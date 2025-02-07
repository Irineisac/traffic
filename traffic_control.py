from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Define input images for each lane
lane_images = ["lane1.jpg", "lane2.jpg", "lane3.jpg", "lane4.jpg"]

# Define vehicle class IDs for detection (Car, Bus, Truck, Motorcycle)
vehicle_classes = [2, 3, 5, 7]  # YOLO class IDs

# Dictionary to store vehicle counts per lane
vehicle_counts = {}

# Process each lane image
for idx, img_path in enumerate(lane_images):
    img = cv2.imread(img_path)  # Load the image

    if img is None:
        print(f"❌ Error: Could not load {img_path}")
        continue  # Skip to the next image if loading fails

    # Run YOLOv8 inference
    results = model(img)

    # Count vehicles in the lane
    vehicle_count = sum(1 for r in results[0].boxes.cls if int(r) in vehicle_classes)

    # Store count
    vehicle_counts[f"Lane {idx+1}"] = vehicle_count

    # Display the annotated image for each lane
    annotated_img = results[0].plot()
    cv2.imshow(f"Lane {idx+1}", annotated_img)
    cv2.waitKey(1000)  # Show each image for 1 second

cv2.destroyAllWindows()

# Print vehicle counts for all lanes
print("\n🚗 **Vehicle counts per lane:**")
for lane, count in vehicle_counts.items():
    print(f"{lane}: {count} vehicles")

# Determine which lane gets the green light (highest traffic)
max_lane = max(vehicle_counts, key=vehicle_counts.get)
print(f"\n🚦 **Green light given to:** {max_lane}")

# Show traffic signal decision
for lane, count in vehicle_counts.items():
    status = "🟢 GREEN" if lane == max_lane else "🔴 RED"
    print(f"{lane}: {count} vehicles → {status}")
