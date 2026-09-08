"""Publish the xArm end-effector pose and twist at the driver's full rate.

The xarm driver reports joint state but never the Cartesian velocity of
the tool: `RobotMsg` carries `pose` (position only) and the SDK's
`rt_tcp_spd` is a scalar magnitude that never reaches ROS at all. Data
collection and VLA inference both need the full 6-DOF twist, so this node
derives it from the joint state with the manipulator Jacobian.

Why the Jacobian rather than differentiating the reported pose:

  * `/xarm/joint_states` already carries measured joint velocities at
    ~100 Hz, so the twist is one linear transform away -- no differencing,
    no filter, and therefore no added lag.
  * `RobotMsg.pose` arrives at 5 Hz on the default `report_type: normal`,
    and its orientation is Euler angles. Differencing those introduces
    wrap-around and gimbal artifacts exactly where a demonstration is most
    interesting -- during fast reorientation.

Everything is reported at the TCP, not the flange. The teleop node drives
the arm with `vc_set_cartesian_velocity`, which acts at the TCP, so
observing anywhere else would mean the recorded action and the recorded
observation refer to different points on the robot.
"""

import math
import threading

import PyKDL
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF
from xarm_msgs.msg import RobotMsg

# RobotMsg.offset is in millimetres; the rest of ROS is in metres.
MM_PER_M = 1000.0


def _kdl_tree_from_urdf(model):
    """Build a KDL tree from a parsed URDF model.

    kdl_parser_py is not packaged in this environment, so the conversion
    is done here. Only the joint types an xArm actually uses are handled;
    anything else becomes a fixed joint, which would show up immediately
    as a wrong DOF count rather than as silently wrong kinematics.
    """
    tree = PyKDL.Tree(model.get_root())

    def add_children(parent):
        for joint in model.joints:
            if joint.parent != parent:
                continue
            kdl_joint = _kdl_joint(joint)
            kdl_frame = _kdl_frame(joint.origin)
            # The xArm URDF carries no <inertial> on link_eef, and inertia
            # is irrelevant to FK and the Jacobian, so a default segment
            # inertia is fine here.
            segment = PyKDL.Segment(joint.child, kdl_joint, kdl_frame)
            tree.addSegment(segment, parent)
            add_children(joint.child)

    add_children(model.get_root())
    return tree


def _kdl_joint(joint):
    origin = _kdl_frame(joint.origin)
    if joint.type in ('revolute', 'continuous'):
        axis = PyKDL.Vector(*(joint.axis or [1.0, 0.0, 0.0]))
        return PyKDL.Joint(joint.name, origin.p, origin.M * axis,
                           PyKDL.Joint.RotAxis)
    if joint.type == 'prismatic':
        axis = PyKDL.Vector(*(joint.axis or [1.0, 0.0, 0.0]))
        return PyKDL.Joint(joint.name, origin.p, origin.M * axis,
                           PyKDL.Joint.TransAxis)
    return PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)


def _kdl_frame(origin):
    if origin is None:
        return PyKDL.Frame()
    xyz = origin.xyz or [0.0, 0.0, 0.0]
    rpy = origin.rpy or [0.0, 0.0, 0.0]
    return PyKDL.Frame(PyKDL.Rotation.RPY(*rpy), PyKDL.Vector(*xyz))


class EefStatePublisher(Node):

    def __init__(self):
        super().__init__('eef_state_publisher')

        self._declare_parameters()
        self._load_parameters()

        # Guards the TCP offset, which the robot_states callback rewrites
        # while the joint_states callback reads it.
        self._offset_lock = threading.Lock()
        self._tcp_offset = None      # PyKDL.Frame, flange -> TCP
        self._offset_logged = False

        self._build_chain()

        # Matches the driver's own joint_states publisher: this is a
        # fixed-rate stream where the freshest sample is what matters.
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pose_pub = self.create_publisher(
            PoseStamped, self.pose_topic, sensor_qos)
        self.twist_pub = self.create_publisher(
            TwistStamped, self.twist_topic, sensor_qos)

        self.create_subscription(
            JointState, self.joint_states_topic, self._on_joint_state, sensor_qos)

        if self.use_reported_offset:
            # robot_states is RELIABLE on the driver side and only 5 Hz,
            # but the offset is a constant -- one message is enough, and
            # a late one simply corrects the assumption.
            self.create_subscription(
                RobotMsg, self.robot_states_topic, self._on_robot_state, 10)

        self.get_logger().info(
            f'publishing TCP state from {self.joint_states_topic}: '
            f'{self.pose_topic}, {self.twist_topic}')

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('joint_states_topic', '/xarm/joint_states')
        self.declare_parameter('robot_states_topic', '/xarm/robot_states')
        self.declare_parameter('pose_topic', '/observation/eef_pose')
        self.declare_parameter('twist_topic', '/observation/eef_twist')

        self.declare_parameter('base_link', 'link_base')
        # The URDF ends at the flange; the TCP is an offset from it that
        # only the controller knows, so it is applied on top of the chain.
        self.declare_parameter('flange_link', 'link_eef')
        # Names the frame the output is expressed in. Not link_base once
        # base_yaw_offset_deg is non-zero -- the controller's base frame
        # has no URDF link, so it gets its own name rather than borrowing
        # one that /tf would resolve to a different orientation.
        self.declare_parameter('frame_id', 'xarm_base')

        # The controller's own base frame is rotated with respect to the
        # URDF's link_base: the same arm pose reads as x=+248mm here and
        # y=-248mm in RobotMsg.pose, with a matching -90 deg of yaw. That
        # is a fixed frame difference, not an error, but the recorded
        # observation has to agree with whatever frame the rest of the
        # cell reasons in -- so this rotation puts the output in the
        # controller's frame, the one RobotMsg.pose and the teleop
        # velocity command already use.
        #
        # Set to 0.0 to publish in the URDF link_base frame instead, which
        # is what /tf and the camera extrinsics use.
        self.declare_parameter('base_yaw_offset_deg', -90.0)

        # Take the offset the controller reports rather than trusting a
        # config value that can silently drift out of sync with what the
        # arm is actually using. The static value is the fallback, and is
        # what the node assumes until the first robot_states arrives.
        self.declare_parameter('use_reported_offset', True)
        self.declare_parameter('tcp_offset', [0.0, 0.0, 185.0, 0.0, 0.0, 0.0])
        self.declare_parameter('tcp_offset_units', 'mm')

    def _load_parameters(self):
        get = self.get_parameter
        self.joint_states_topic = get('joint_states_topic').value
        self.robot_states_topic = get('robot_states_topic').value
        self.pose_topic = get('pose_topic').value
        self.twist_topic = get('twist_topic').value

        self.base_link = get('base_link').value
        self.flange_link = get('flange_link').value
        self.frame_id = get('frame_id').value

        yaw_deg = float(get('base_yaw_offset_deg').value)
        self._base_rot = PyKDL.Rotation.RotZ(math.radians(yaw_deg))
        self._base_yaw_deg = yaw_deg

        self.use_reported_offset = bool(get('use_reported_offset').value)

        units = str(get('tcp_offset_units').value).strip().lower()
        if units not in ('mm', 'm'):
            self.get_logger().warn(
                f"unknown tcp_offset_units '{units}', assuming 'mm'")
            units = 'mm'
        scale = 1.0 / MM_PER_M if units == 'mm' else 1.0

        raw = [float(v) for v in get('tcp_offset').value]
        if len(raw) != 6:
            self.get_logger().error(
                f'tcp_offset needs 6 values, got {len(raw)}; using zero offset')
            raw = [0.0] * 6
        self._static_offset = self._offset_frame(
            [v * scale for v in raw[:3]], raw[3:])

    @staticmethod
    def _offset_frame(xyz_m, rpy_rad):
        return PyKDL.Frame(PyKDL.Rotation.RPY(*rpy_rad), PyKDL.Vector(*xyz_m))

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _build_chain(self):
        """Fetch the URDF from robot_state_publisher and build the chain.

        The description is read from the other node's parameter rather
        than from a file so this node always matches the model the rest
        of the stack is using.
        """
        urdf_xml = self._fetch_robot_description()
        model = URDF.from_xml_string(urdf_xml)
        tree = _kdl_tree_from_urdf(model)

        self.chain = tree.getChain(self.base_link, self.flange_link)
        self.dof = self.chain.getNrOfJoints()
        if self.dof == 0:
            raise RuntimeError(
                f'no movable joints between {self.base_link} and '
                f'{self.flange_link} -- check base_link/flange_link')

        # Joint order along the chain, used to reorder incoming
        # JointState by name: the driver's ordering is not guaranteed to
        # match the URDF's, and a silent mismatch produces a plausible
        # but wrong twist.
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getType() != PyKDL.Joint.Fixed:
                self.joint_names.append(joint.getName())

        self._fk = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self._jac = PyKDL.ChainJntToJacSolver(self.chain)
        self._q = PyKDL.JntArray(self.dof)
        self._jacobian = PyKDL.Jacobian(self.dof)

        self.get_logger().info(
            f'chain {self.base_link} -> {self.flange_link}: '
            f'{self.dof} joints {self.joint_names}')

    def _fetch_robot_description(self):
        """Read robot_description from robot_state_publisher.

        Blocks until it is available: without the model there is nothing
        this node can compute, and coming up before the description is
        published is normal when everything launches together.
        """
        client = self.create_client(
            GetParameters, '/robot_state_publisher/get_parameters')
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                'waiting for /robot_state_publisher to provide robot_description ...')

        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        values = future.result().values
        if not values or not values[0].string_value:
            raise RuntimeError('robot_description is empty')
        return values[0].string_value

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_robot_state(self, msg):
        """Adopt the TCP offset the controller reports."""
        offset = list(msg.offset)
        if len(offset) < 6:
            return
        frame = self._offset_frame(
            [v / MM_PER_M for v in offset[:3]], offset[3:])
        with self._offset_lock:
            self._tcp_offset = frame
        if not self._offset_logged:
            self._offset_logged = True
            self.get_logger().info(
                f'TCP offset from controller: '
                f'xyz={[round(v, 4) for v in offset[:3]]} mm '
                f'rpy={[round(v, 4) for v in offset[3:]]} rad')

    def _on_joint_state(self, msg):
        if not self._fill_joints(msg):
            return

        with self._offset_lock:
            offset = self._tcp_offset
        if offset is None:
            # No controller report yet: fall back to the configured value
            # so the node is useful from the first message.
            offset = self._static_offset

        flange = PyKDL.Frame()
        if self._fk.JntToCart(self._q, flange) < 0:
            self.get_logger().warn('FK failed', throttle_duration_sec=5.0)
            return
        if self._jac.JntToJac(self._q, self._jacobian) < 0:
            self.get_logger().warn('Jacobian failed', throttle_duration_sec=5.0)
            return

        tcp = flange * offset
        twist = self._tcp_twist(msg, flange, offset)
        if twist is None:
            return

        self._publish(msg.header.stamp, tcp, twist)

    def _fill_joints(self, msg):
        """Copy positions into the KDL array in chain order.

        Returns False when the message does not carry every joint of the
        chain, which happens while a partial publisher is still coming up.
        """
        index = {name: i for i, name in enumerate(msg.name)}
        for i, name in enumerate(self.joint_names):
            j = index.get(name)
            if j is None or j >= len(msg.position):
                self.get_logger().warn(
                    f'joint {name} missing from {self.joint_states_topic}',
                    throttle_duration_sec=5.0)
                return False
            self._q[i] = msg.position[j]
        return True

    def _tcp_twist(self, msg, flange, offset):
        """Twist at the TCP, in the base frame.

        The Jacobian is taken at the flange, so the linear part has to be
        moved to the TCP: a rigid body rotating at omega about the flange
        carries the TCP along at an extra omega x r. Angular velocity is
        the same everywhere on a rigid body, so it transfers unchanged.
        """
        index = {name: i for i, name in enumerate(msg.name)}
        if len(msg.velocity) == 0:
            self.get_logger().warn(
                f'{self.joint_states_topic} carries no velocities -- '
                f'the driver reports them only on firmware >= 1.8.103',
                throttle_duration_sec=10.0)
            return None

        qdot = [0.0] * self.dof
        for i, name in enumerate(self.joint_names):
            j = index.get(name)
            if j is None or j >= len(msg.velocity):
                return None
            qdot[i] = msg.velocity[j]

        linear = PyKDL.Vector()
        angular = PyKDL.Vector()
        for row, target in ((0, linear), (3, angular)):
            for axis in range(3):
                total = 0.0
                for col in range(self.dof):
                    total += self._jacobian[row + axis, col] * qdot[col]
                target[axis] = total

        # r is the flange->TCP lever arm expressed in the base frame.
        lever = flange.M * offset.p
        linear = linear + angular * lever
        return linear, angular

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, stamp, tcp, twist):
        linear, angular = twist

        # Rotate out of the URDF base frame into the controller's. A pure
        # rotation about the shared origin, so the position rotates, the
        # orientation is pre-multiplied, and BOTH halves of the twist go
        # with them -- rotating only the linear part would leave the
        # angular velocity describing a different frame than the pose it
        # accompanies.
        if self._base_yaw_deg:
            tcp = PyKDL.Frame(self._base_rot) * tcp
            linear = self._base_rot * linear
            angular = self._base_rot * angular

        pose = PoseStamped()
        # Reuse the driver's stamp rather than "now": this state describes
        # the instant the arm was sampled, and every other recorded stream
        # is aligned against that.
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = tcp.p[0]
        pose.pose.position.y = tcp.p[1]
        pose.pose.position.z = tcp.p[2]
        qx, qy, qz, qw = tcp.M.GetQuaternion()
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.pose_pub.publish(pose)

        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = linear[0]
        msg.twist.linear.y = linear[1]
        msg.twist.linear.z = linear[2]
        msg.twist.angular.x = angular[0]
        msg.twist.angular.y = angular[1]
        msg.twist.angular.z = angular[2]
        self.twist_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EefStatePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
