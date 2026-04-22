import { configureStore } from '@reduxjs/toolkit'
import rosReducer from '../features/ros/rosSlice'
import inferenceReducer from '../features/inference/inferenceSlice'

export const store = configureStore({
  reducer: {
    ros: rosReducer,
    inference: inferenceReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({ serializableCheck: false }),
})
