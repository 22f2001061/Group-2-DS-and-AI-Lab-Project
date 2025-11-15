#!/bin/bash

# Vision Assistance Server Deployment Script
# 
# USAGE:
#   1. Upload this script along with your project files to your VM
#   2. Run from your project directory containing:
#      - vision_assist_server.py
#      - yolov8n_optuna_best.pt
#      - deploy.sh (this script)
#   3. Execute: chmod +x deploy.sh && ./deploy.sh
#
# This script will:
#   - Install all system dependencies
#   - Create Python virtual environment
#   - Install Python packages
#   - Copy files to /opt/vision-assist/
#   - Create systemd service
#   - Start the service automatically

echo "🚀 Vision Assistance Server - VM Deployment Script"
echo "=================================================="

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip if not already installed
echo "🐍 Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies for OpenCV and audio
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    portaudio19-dev \
    python3-dev \
    pkg-config

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p /opt/vision-assist
sudo chown $USER:$USER /opt/vision-assist
cd /opt/vision-assist

# Create Python virtual environment
echo "🌐 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "ultralytics>=8.0.0"
pip install "opencv-python-headless>=4.5.0"  # Use headless version for servers
pip install "numpy>=1.21.0"
pip install "gtts>=2.2.0"
pip install "pydub>=0.25.0"
pip install "fastapi>=0.100.0"
pip install "uvicorn[standard]>=0.20.0"  # Standard includes WebSocket support
pip install "websockets>=10.0"  # Additional WebSocket support
pip install "python-multipart"  # Required for FastAPI file uploads
pip install "Pillow>=8.0.0"
pip install "requests>=2.25.0"

echo "✅ Dependencies installed successfully!"

# Verify FFmpeg tools are available
echo "🔍 Verifying FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg: $(which ffmpeg)"
else
    echo "❌ ffmpeg not found in PATH"
fi

if command -v ffprobe &> /dev/null; then
    echo "✅ ffprobe: $(which ffprobe)"
else
    echo "❌ ffprobe not found in PATH"
    # Try installing ffmpeg tools explicitly
    echo "🔧 Installing ffmpeg tools explicitly..."
    sudo apt install -y ffmpeg
fi

# Create systemd service file
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/vision-assist.service > /dev/null <<EOF
[Unit]
Description=Vision Assistance Server with WebSocket Support
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/vision-assist
Environment=PATH=/opt/vision-assist/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/vision-assist/venv/bin/uvicorn vision_assist_server:app --host 0.0.0.0 --port 80 --log-level info
Restart=always
RestartSec=3
KillMode=mixed
KillSignal=SIGINT
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set up firewall rules (if UFW is installed)
if command -v ufw &> /dev/null; then
    echo "🔥 Configuring firewall..."
    sudo ufw allow 80/tcp
    echo "Port 80 opened in firewall"
fi

# Go back to the original directory where script was called from
cd "$OLDPWD" 2>/dev/null || cd ~

# Check if this script is being run from the project directory
if [ -f "vision_assist_server.py" ] && [ -f "yolov8n_optuna_best.pt" ]; then
    echo "📋 Copying application files..."
    cp vision_assist_server.py /opt/vision-assist/
    cp yolov8n_optuna_best.pt /opt/vision-assist/
    cp requirements.txt /opt/vision-assist/ 2>/dev/null || echo "requirements.txt not found, continuing..."
    
    # Copy WebSocket client files if they exist
    if [ -f "websocket_camera_client.html" ]; then
        cp websocket_camera_client.html /opt/vision-assist/
        echo "✅ WebSocket HTML client copied"
    fi
    
    if [ -f "websocket_live_test.py" ]; then
        cp websocket_live_test.py /opt/vision-assist/
        echo "✅ WebSocket test client copied"
    fi
    
    if [ -f "WEBSOCKET_STREAMING.md" ]; then
        cp WEBSOCKET_STREAMING.md /opt/vision-assist/
        echo "✅ WebSocket documentation copied"
    fi
    
    # Fix file ownership
    chown $USER:$USER /opt/vision-assist/vision_assist_server.py /opt/vision-assist/yolov8n_optuna_best.pt
    chown $USER:$USER /opt/vision-assist/*.html /opt/vision-assist/*.py /opt/vision-assist/*.md 2>/dev/null || true
    
    # Update model path in the server file to use the correct location
    sed -i "s|YOLO_MODEL_PATH = './yolov8n_optuna_best.pt'|YOLO_MODEL_PATH = '/opt/vision-assist/yolov8n_optuna_best.pt'|" /opt/vision-assist/vision_assist_server.py
    
    # Create a simple nginx configuration for serving static files (optional)
    if command -v nginx &> /dev/null; then
        echo "🌐 Configuring nginx for static file serving..."
        sudo mkdir -p /var/www/vision-assist
        sudo cp /opt/vision-assist/*.html /var/www/vision-assist/ 2>/dev/null || true
        sudo cp /opt/vision-assist/*.md /var/www/vision-assist/ 2>/dev/null || true
    fi
    
    echo "✅ Application files copied and configured!"
    
    # Verify files are in place
    echo "📋 Files in /opt/vision-assist/:"
    ls -la /opt/vision-assist/ | grep -v "venv"
else
    echo "⚠️  Could not find required files in current directory: $(pwd)"
    echo "   Looking for:"
    echo "   - vision_assist_server.py"
    echo "   - yolov8n_optuna_best.pt"
    echo ""
    echo "   Please copy these files manually:"
    echo "   sudo cp vision_assist_server.py /opt/vision-assist/"
    echo "   sudo cp yolov8n_optuna_best.pt /opt/vision-assist/"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "🚀 Starting the service:"
echo "Enabling and starting vision-assist service..."
sudo systemctl daemon-reload
sudo systemctl enable vision-assist
sudo systemctl start vision-assist

# Wait a moment for service to start
sleep 3

echo ""
echo "📊 Service Status:"
sudo systemctl status vision-assist --no-pager -l

echo ""
echo "🔍 Verifying deployment..."
echo "Testing server health..."
health_response=$(curl -s http://localhost/health 2>/dev/null || echo "failed")
if [[ $health_response == *"healthy"* ]]; then
    echo "✅ Health check: PASSED"
else
    echo "⚠️  Health check: FAILED - Service may still be starting"
fi

echo ""
echo "✅ Deployment Summary:"
echo "📁 Application directory: /opt/vision-assist"
echo "🔗 Server URL: http://$(hostname -I | awk '{print $1}'):80"
echo "🔗 Public URL: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR-PUBLIC-IP'):80"
echo "🔧 Service management:"
echo "   • Status: sudo systemctl status vision-assist"
echo "   • Stop: sudo systemctl stop vision-assist"
echo "   • Start: sudo systemctl start vision-assist"
echo "   • Restart: sudo systemctl restart vision-assist"
echo "   • Logs: sudo journalctl -u vision-assist -f"
echo ""
echo "🔬 Test the deployment:"
echo "   • Health: curl http://localhost/health"
echo "   • Cameras: curl http://localhost/cameras"
echo "   • WebSocket Status: curl http://localhost/ws/status"
echo "   • Upload image: curl -X POST -F \"file=@image.jpg\" http://localhost/process_frame"
echo "   • WebSocket Client: http://$(hostname -I | awk '{print $1}'):80/websocket_camera_client.html"
echo ""
echo "📱 WebSocket Testing:"
echo "   • Python test: cd /opt/vision-assist && ./venv/bin/python websocket_live_test.py"
echo "   • HTML client: Open websocket_camera_client.html in browser"
echo "   • WebSocket URL: ws://$(hostname -I | awk '{print $1}'):80/ws/camera/stream"
echo ""
echo "🎆 Vision Assistance Server with Real-Time WebSocket Streaming is ready!"
echo "🎧 Features enabled: Object detection, Audio alerts, Continuous narration, WebSocket streaming"