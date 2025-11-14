from ultralytics import YOLO
import cv2
import smtplib
import threading
import pygame
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ---------------- Email credentials ----------------
fromaddr = "agaash1809@gmail.com"
toaddr = "727722eucs009@skcet.ac.in"
app_password = "xsxsrrixdcplmjvg"

# ---------------- YOLO model ----------------
model = YOLO("model/weapon_model.pt")
CONFIDENCE_THRESHOLD = 0.5

# ---------------- Load single image ----------------
image_path = r"input images\weapon (78).jpg"
frame = cv2.imread(image_path)

# ---------------- Alarm control ----------------
alarm_active = False

def play_buzzer():
    """Play custom alarm sound continuously in a separate thread"""
    global alarm_active
    pygame.mixer.init()
    pygame.mixer.music.load("alert2.mp3")  # <-- Your custom buzzer sound file
    pygame.mixer.music.play(-1)           # Loop continuously
    while alarm_active:
        continue
    pygame.mixer.music.stop()

# ---------------- Email function ----------------
def mail(text):
    """Send email with detection alert and image"""
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "⚠️ WEAPON DETECTION ALERT ⚠️"
    msg.attach(MIMEText(text, 'plain'))

    filename = "mail/weapon.jpg"
    with open(filename, "rb") as attachment:
        p = MIMEBase('application', 'octet-stream')
        p.set_payload(attachment.read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', f"attachment; filename={filename}")
        msg.attach(p)

    try:
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(fromaddr, app_password)
        s.send_message(msg)
        s.quit()
        print(" ⚠️⚠️⚠️ Weapon detected! ⚠️⚠️⚠️\n Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# ---------------- Run detection ----------------
results = model(frame, conf=CONFIDENCE_THRESHOLD)
weapon_detected = False

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0,  255), 2)

        weapon_detected = True

        # Save frame & send email
        cv2.imwrite('mail/weapon.jpg', frame)
        mail("⚠️ WEAPON DETECTED! Please check immediately.")

        # Start alarm thread if not already active
        if not alarm_active:
            alarm_active = True
            threading.Thread(target=play_buzzer, daemon=True).start()

if not weapon_detected:
    print("NO WEAPON DETECTED")

# ---------------- Display detection window ----------------
while True:
    cv2.imshow("Weapon Detection", frame)
    key = cv2.waitKey(1)
    if key != -1:  # Any key pressed stops alarm & window
        alarm_active = False
        break

cv2.destroyAllWindows()
