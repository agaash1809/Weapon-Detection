import cv2
from ultralytics import YOLO
import smtplib
import threading
import pygame
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ---------------- Email Setup ----------------
fromaddr = "agaash1809@gmail.com"
toaddr = "727722eucs009@skcet.ac.in"
app_password = "xsxsrrixdcplmjvg"  # your app password

# ---------------- YOLO Model ----------------
weapon_model = YOLO('model/weapon_model.pt')
weapon_names = weapon_model.names

# ---------------- Video Source (0 = webcam) ----------------
cap = cv2.VideoCapture(0)

# FPS setup
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30  # ms

# Confidence threshold
weapon_conf = 0.3

# ---------------- Alarm Control ----------------
alarm_active = False

def play_buzzer():
    """Play continuous buzzer sound using a custom alert.wav file"""
    global alarm_active
    pygame.mixer.init()
    pygame.mixer.music.load("alert2.mp3")  # <-- Put your custom buzzer sound file here
    pygame.mixer.music.play(-1)           # Loop indefinitely
    while alarm_active:
        continue
    pygame.mixer.music.stop()

# ---------------- Mail Function ----------------
def mail(text):
    """Send email with detected weapon image"""
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
        print("⚠️ Weapon detected — Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO weapon detection
    results_weapon = weapon_model.track(frame, conf=weapon_conf, persist=True, verbose=False)

    # Process weapon detections
    weapon_detected = False
    if results_weapon and len(results_weapon[0].boxes) > 0:
        boxes = results_weapon[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results_weapon[0].boxes.cls.int().cpu().tolist()
        confs = results_weapon[0].boxes.conf.cpu().numpy().tolist()

        for box, class_id, conf in zip(boxes, class_ids, confs):
            x1, y1, x2, y2 = box
            name = weapon_names[class_id].lower()

            if name in ["gun", "knife"]:
                weapon_detected = True
                print(f"Weapon Detected: {name.capitalize()}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name.capitalize(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite('mail/weapon.jpg', frame)
                mail("⚠️ WEAPON DETECTED!\nPlease check immediately.")

                # Start alarm thread
                if not alarm_active:
                    alarm_active = True
                    threading.Thread(target=play_buzzer, daemon=True).start()

    # If no weapon detected, stop alarm
    if not weapon_detected and alarm_active:
        alarm_active = False

    # Display video
    cv2.imshow("Weapon Detection", frame)

    # Press 'q' to quit and stop alarm
    if cv2.waitKey(1) & 0xFF == ord("q"):
        alarm_active = False
        break

cap.release()
cv2.destroyAllWindows()
