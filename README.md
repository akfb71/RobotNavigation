# **Neural Network for Collision Prediction**

## **Overview**

This project explores the use of deep learning for predictive collision avoidance in autonomous robotic navigation.  
It integrates a simulated robot environment with a neural network trained to predict whether a robot will collide with obstacles based on its sensor readings and movement actions.

The robot navigates a 2D environment using a physics engine and collects real-time data during exploration. A neural network trained on this data enables the robot to evaluate potential actions before taking them, allowing for safe and intelligent navigation decisions.

---

## **Objectives**

- Collect and manage datasets consisting of distance sensor readings, steering actions, and collision outcomes.  

---

## **Technology Stack**

- **Operating System:** macOS Sonoma or similar version
- **Language:** Python 3.12  
- **Deep Learning Framework:** PyTorch  
- **Simulation & Visualization:** Pymunk, Pygame
- **Data Processing & ML Tools:** NumPy, SciPy, scikit-learn  
- **Visualization:** Matplotlib  
- **Noise Generation:** Perlin noise for random but smooth steering control

---

## **Installation and Setup**

### **1. Clone the Repository**

```
git clone https://github.com/fatimabasheer/RobotNavigation.git
cd RobotNavigation
```

### **2. Create Virtual environment

```
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### **3. Install dependencies

```
pip install -r requirements.txt
```

## Part 1 : Data Collection

The data collection simulates robot in wandering randomly in a 2D world. The collected data is in submissions.csv.
Note : In this phase the robot doesn't avoid collisions as we want to gather diverse sensor-action-outcome samples for supervised learning.

Each sample records:
Five distance sensor readings
The steering action taken
A collision flag (0 = no collision, 1 = collision)


