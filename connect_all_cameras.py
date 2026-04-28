#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import time
from inference import classify_all_arms, send_alert

# ====================== INSTÄLLNINGAR ======================
NUM_CAMERAS = 12  # Ändra till 9 eller 12 beroende på robot
OUTPUT_DIR = "camera2_images"
bridge = CvBridge()

# En dictionary som håller senaste bilden från varje kamera
latest_images = {}

# ====================== CALLBACK FÖR VARJE KAMERA ======================
def make_callback(camera_id):
    """
    Skapar en callback-funktion för varje kamera.
    camera_id är kamerans nummer, t.ex. 1, 2, 3...
    """
    def callback(msg):
        try:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            latest_images[camera_id] = image
        except Exception as e:
            rospy.logerr(f"Fel vid kamera {camera_id}: {e}")
    return callback

# ====================== SPARA BILDER FRÅN ALLA KAMEROR ======================
def save_all_images():
    """
    Sparar en bild från varje kamera till rätt mapp
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Rensa gamla bilder innan nya sparas
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            os.remove(os.path.join(OUTPUT_DIR, f))
    rospy.loginfo("Gamla bilder raderade")

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    saved = []
    for camera_id in range(1, NUM_CAMERAS + 1):
        if camera_id in latest_images:
            filename = os.path.join(
                OUTPUT_DIR,
                f"camera{camera_id}_{timestamp}.png"
            )
            cv2.imwrite(filename, latest_images[camera_id])
            saved.append(camera_id)
            rospy.loginfo(f"Bild sparad från kamera {camera_id}")
        else:
            rospy.logwarn(f"Ingen bild tillgänglig från kamera {camera_id}")

    return saved

# ====================== HUVUDFUNKTION ======================
def main():
    rospy.init_node('all_cameras_collector', anonymous=True)

    # Prenumerera på alla kameror
    for camera_id in range(1, NUM_CAMERAS + 1):
        topic = f"/camera{camera_id}/color/image_raw"
        rospy.Subscriber(topic, Image, make_callback(camera_id))
        rospy.loginfo(f"Prenumererar på {topic}")

    rospy.loginfo("Väntar på bilder från alla kameror...")

    # Vänta tills vi har bilder från alla kameror
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if len(latest_images) == NUM_CAMERAS:
            rospy.loginfo("Bilder mottagna från alla kameror!")
            break
        rospy.loginfo(
            f"Väntar... {len(latest_images)}/{NUM_CAMERAS} kameror redo"
        )
        rate.sleep()

    # Spara alla bilder
    rospy.loginfo("Sparar bilder från alla kameror...")
    saved_cameras = save_all_images()

    # Kör inferens på alla bilder
    rospy.loginfo("Kör klassificering på alla armar...")
    results = classify_all_arms(OUTPUT_DIR)

    # Hitta skadade armar
    damaged_arms = {
        arm: result for arm, result in results.items()
        if result['class'] != 'intakta'
    }

    # Skicka notifikation om det finns skador
    if damaged_arms:
        rospy.logwarn(f"VARNING: {len(damaged_arms)} skadade tips hittades!")
        send_alert(damaged_arms)
    else:
        rospy.loginfo("Alla tips är intakta!")

    rospy.loginfo("Klar!")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Avslutar...")
    except Exception as e:
        rospy.logerr(f"Fel: {e}")