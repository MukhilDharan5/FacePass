"""Offline fixtures and integration tests; no physical webcam is opened."""

import cv2

# Use the same OpenCV thread budget as the finite capture worker.
cv2.setNumThreads(1)
