# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD functions-framework --target=handle_slack_event --port=${PORT}
