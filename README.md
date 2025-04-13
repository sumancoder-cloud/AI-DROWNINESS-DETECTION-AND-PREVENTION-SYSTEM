# AI Drowsiness Detection and Prevention System

A real-time drowsiness detection system that uses computer vision and machine learning to detect driver drowsiness and prevent accidents. This system continuously monitors the driver's face and eyes to detect signs of drowsiness and provides immediate alerts to prevent potential accidents.

## 🚀 Features

- **Real-time Face Detection**: Uses dlib's facial landmark detector to identify the driver's face
- **Eye Aspect Ratio (EAR) Analysis**: Monitors eye closure duration to detect drowsiness
- **Head Pose Estimation**: Tracks head position to detect when the driver is not looking at the road
- **Multiple Alert Systems**:
  - Audio alerts using custom alarm sounds
  - Voice alerts using text-to-speech
  - SMS notifications to emergency contacts
- **Web Interface**: User-friendly interface for easy interaction and monitoring
- **Emergency Contact System**: Ability to notify emergency contacts via SMS

## 🛠️ Technical Details

### Detection Methods
1. **Eye Aspect Ratio (EAR)**: 
   - Calculates the ratio of eye width to height
   - Triggers alert when eyes are closed for extended periods
   
2. **Head Pose Estimation**:
   - Uses facial landmarks to estimate head orientation
   - Detects when driver is looking away from the road

3. **Alert System**:
   - Audio alerts using pygame
   - Voice alerts using pyttsx3
   - SMS notifications using Twilio API

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam
- Internet connection (for SMS notifications)
- Twilio account (for SMS alerts)

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/sumancoder-cloud/AI-DROWNINESS-DETECTION-AND-PREVENTION-SYSTEM.git
cd AI-DROWNINESS-DETECTION-AND-PREVENTION-SYSTEM
```

2. **Create and activate virtual environment**:
```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file with your Twilio credentials:
```
TWILIO_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
TWILIO_PHONE_NUMBER=your_phone_number_here
```

## 💻 Usage

1. **Start the application**:
```bash
python project.py
```

2. **Access the web interface**:
- Open your web browser
- Go to `http://localhost:5000`

3. **Using the system**:
- Enter your emergency contact number
- Click "Start Detection"
- The system will begin monitoring for drowsiness
- Alerts will be triggered automatically when drowsiness is detected

## 📊 System Architecture

```
├── project.py              # Main application file
├── learning.py            # Machine learning models
├── templates/             # Web interface templates
│   ├── phone.html        # Phone number input page
│   └── detection.html    # Detection monitoring page
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🔧 Configuration

The system can be configured by modifying these parameters in `project.py`:
- `EAR_THRESHOLD`: Eye aspect ratio threshold for drowsiness detection
- `CONSECUTIVE_FRAMES`: Number of consecutive frames for drowsiness confirmation
- `ALERT_DURATION`: Duration of audio alerts
- `SMS_COOLDOWN`: Time between SMS alerts

## 🛡️ Safety Features

- Multiple alert systems for redundancy
- Configurable sensitivity levels
- Emergency contact notification
- Real-time monitoring and feedback

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- dlib library for facial landmark detection
- OpenCV for computer vision capabilities
- Twilio for SMS notification services
- Flask for web interface

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 📈 Future Improvements

- [ ] Mobile app integration
- [ ] Cloud-based monitoring
- [ ] Machine learning model improvements
- [ ] Additional alert methods
- [ ] Dashboard for monitoring multiple drivers
