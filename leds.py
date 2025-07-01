from pi5neo import Pi5Neo

# Initialize the Pi5Neo class with 10 LEDs and an SPI speed of 800kHz
neo = Pi5Neo('/dev/spidev0.0', 144, 800)
neo.fill_strip(0, 0, 0)
neo.update_strip()