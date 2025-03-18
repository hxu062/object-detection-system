FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Download model files
RUN python core/download_models.py

# Set environment variables for QT
ENV QT_DEBUG_PLUGINS=1
ENV QT_X11_NO_MITSHM=1

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]

# Default command shows help
CMD ["--help"] 