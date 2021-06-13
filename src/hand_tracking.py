# -*- coding: utf-8 -*-
"""
Hand Tracing Module
By: Renuá Meireles, Adapted from Murtaza Hassan (https://www.murtazahassan.com)
"""
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands,
            self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # if self.results.multi_hand_landmarks:
        #     for handLms in self.results.multi_hand_landmarks:
        #         if draw:
        #             self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for _id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([_id, cx, cy])
        return self.lmList

    def fingersUp(self):
        # Thumb
        hand = 1 if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1] else 0

        # 5 Fingers
        fingers = [1 if self.lmList[self.tipIds[_id]][2] < self.lmList[self.tipIds[_id] - 2][2] else 0 for _id in range(5)]
        return fingers, hand

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        key=cv2.waitKey(1)
        if key == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Bye!')
        exit(1)