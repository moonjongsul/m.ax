import { configureStore } from '@reduxjs/toolkit'
import rosReducer from '../features/ros/rosSlice'
import inferenceReducer from '../features/inference/inferenceSlice'
import telemetryReducer from '../features/telemetry/telemetrySlice'

export const store = configureStore({
  reducer: {
    ros: rosReducer,
    inference: inferenceReducer,
    telemetry: telemetryReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({ serializableCheck: false }),
})
