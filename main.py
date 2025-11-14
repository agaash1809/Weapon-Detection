import cv2
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
    
fromaddr = "agaash1809@gmail.com"
toaddr ="727722eucs009@skcet.ac.in"
# Load YOLOv8 weapon detection model
weapon_model = YOLO('model/weapon_model.pt')
weapon_names = weapon_model.names

# Load video
# cap = cv2.VideoCapture("input videos/input1.mp4")
cap = cv2.VideoCapture(0)

# Get original video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30  # ms

# Confidence threshold
weapon_conf = 0.3   

def mail(text):
    print(text)
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "WEAPON DETECTION SYSTEM"
    body = text
    msg.attach(MIMEText(body, 'plain'))
    filename = "mail/weapon.jpg"
    attachment = open("mail/weapon.jpg", "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload(attachment.read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', f"attachment; filename={filename}")
    msg.attach(p)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr, "xsxsrrixdcplmjvg")
    text = msg.as_string()
    s.sendmail(fromaddr, toaddr, text)
    s.quit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Run weapon detection
    results_weapon = weapon_model.track(frame, conf=weapon_conf, persist=True, verbose=False)

    # Process weapon detections
    if results_weapon and len(results_weapon[0].boxes) > 0:
        boxes = results_weapon[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results_weapon[0].boxes.cls.int().cpu().tolist()
        confs = results_weapon[0].boxes.conf.cpu().numpy().tolist()

        for box, class_id, conf in zip(boxes, class_ids, confs):
            x1, y1, x2, y2 = box
            name = weapon_names[class_id].lower()

            if name in ["gun", "knife"]:   # adjust if your dataset has different names
                print("Weapon Detected:", name.capitalize())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name.capitalize(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(f'mail/weapon.jpg', frame)
                mail("WEAPON DETECTED \n Please Check It")

    # Show frame
    cv2.imshow("Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
