# yellowblock_obstacle_detection# Yellow Block Obstacle Detection

This project is designed to detect obstacles on yellow tactile paving blocks using YOLOv8 models. The project is divided into two main steps: detecting the yellow blocks and then detecting obstacles on those blocks.

## Installation

Before running the project, you need to install the required libraries. This project uses Python and several external libraries, including OpenCV and the `ultralytics` YOLO library.

1. **Clone the repository:**

   git clone https://github.com/swnswx/yellowblock_obstacle_detection.git
   
   cd yellowblock_obstacle_detection

3. **Create a virtual environment (optional but recommended):**

    python -m venv venv
   
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

5. **Install the required libraries:**

    pip install opencv-python-headless ultralytics numpy
