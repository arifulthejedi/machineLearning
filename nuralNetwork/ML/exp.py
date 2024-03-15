from PIL import Image

def detect_color_mode(image_path):
    try:
        img = Image.open(image_path)
        mode = img.mode
        if mode == "CMYK":
            print("Image is in CMYK color mode.")
        elif mode == "RGB":
            print("Image is in RGB color mode.")
        elif mode == "RGBA":
            print("Image is in RGBA color mode.")
        else:
            print("Image is in color mode:", mode)
    except FileNotFoundError:
        print("Error: File not found.")
    except PermissionError:
        print("Error: Permission denied to access the file.")
    except Exception as e:
        print("Error:", e)

# Provide the path to your image file
image_path = "src1.png"  # Make sure your image is in PNG format
detect_color_mode(image_path)


from PIL import Image

def rgba_to_rgb(image_path):
    try:
        img = Image.open(image_path)
        mode = img.mode
        if mode == "CMYK":
            print("Image is in CMYK color mode.")
        elif mode == "RGB":
            print("Image is in RGB color mode.")
        elif mode == "RGBA":
            print("Image is in RGBA color mode. Converting to RGB...")
            img = img.convert('RGB')
            print("Image has been converted to RGB color mode.")
        else:
            print("Image is in color mode:", mode)
    except FileNotFoundError:
        print("Error: File not found.")
    except PermissionError:
        print("Error: Permission denied to access the file.")
    except Exception as e:
        print("Error:", e)

# Provide the path to your image file
# image_path = "src1.png"  # Make sure your image is in PNG format
# rgba_to_rgb(image_path)
