import { configureStore } from '@reduxjs/toolkit'
import rosReducer from '../features/ros/rosSlice'
import inferenceReducer from '../features/inference/inferenceSlice'
import telemetryReducer from '../features/telemetry/telemetrySlice'
import lerobotReducer from '../features/lerobot/lerobotSlice'

export const store = configureStore({
  reducer: {
    ros: rosReducer,
    inference: inferenceReducer,
    telemetry: telemetryReducer,
    lerobot: lerobotReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({ serializableCheck: false }),
})
