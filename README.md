# Gesture Drawing App

A computer vision-based drawing application that allows users to draw on a virtual canvas using hand gestures. The app uses MediaPipe for hand tracking and OpenCV for image processing and visualization.

## Features

- Hand gesture-based drawing interface
- Multiple drawing tools and colors
- Scrollable canvas
- Interactive toolbar with hover selection
- Main menu interface
- Real-time hand tracking and gesture recognition

## Requirements

```
opencv-python==4.10.0.84
mediapipe==0.10.18
numpy==1.26.4
```

## Installation

1. Clone the repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python gesture_drawing.py
```

### Controls

- **Menu Mode:**
  - Hover over buttons for 2 seconds to select
  - Options include "Start Drawing" and "Exit"

- **Drawing Mode:**
  - Open hand: Draw on canvas
  - Pinch gesture: Stop drawing
  - Fist gesture: Enable scroll mode
  - Hover over toolbar items for 2 seconds to select tools

### Tools and Features

- **Colors:**
  - Red
  - Green
  - Black
  - Magenta
  - Yellow

- **Drawing Sizes:**
  - Size 5
  - Size 10

- **Tools:**
  - Pen
  - Eraser
  - Back (returns to menu)

## Configuration

Key parameters can be adjusted in the configuration section of the code:

- `CAM_WIDTH`: Camera width (default: 1280)
- `CAM_HEIGHT`: Camera height (default: 720)
- `PINCH_THRESHOLD`: Distance threshold for pinch detection
- `FIST_THRESHOLD`: Distance threshold for fist detection
- `SCROLL_SPEED`: Scrolling speed
- `HOVER_SELECT_TIME`: Time required to hover for selection

## System Requirements

- Python 3.10-3.12
- Webcam
- Sufficient lighting for hand detection

## Exit Application

- Press 'q' to quit
- Or select "Exit" from the main menu

## Troubleshooting

If you encounter issues:

1. Ensure proper lighting for hand detection
2. Check if your webcam is properly connected and accessible
3. Verify Python version compatibility
4. Make sure all dependencies are correctly installed

## Note

The application uses your computer's webcam for hand tracking. Ensure your webcam is working and properly configured before running the application.