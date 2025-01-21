import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def unique_contours(mask):
    if isinstance(mask, str):
        return unique_contours_helper(cv2.imread(mask))
    else:
        return unique_contours_helper(mask)

def unique_contours_helper(mask_image):
    #####  preprocessing based on https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/
    image = mask_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    canny = cv2.Canny(gray, 30, 80, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)
    
    (contours, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #####
    unique_contours = {}
    for c in contours:
        keyset = set(unique_contours.keys())
        moments = cv2.moments(c)
        if (moments['m00'] > 3): 
            x = int(moments['m10']/moments['m00']) 
            y = int(moments['m01']/moments['m00']) 

            duplicate = False
            # this method better if
            #n^2/2 > 64n -> n > 128
            for dx in range(-8,8):
                for dy in range(-8,8):
                    if (x + dx, y + dy) in keyset:
                        duplicate=True
            if not duplicate:
                unique_contours[(x,y)] = (math.sqrt(cv2.contourArea(c)/math.pi), cv2.contourArea(c))
    # returns unique contours as circles with dict of (centerX, centerY) to (radius, area)
    return unique_contours

def extract_circles(estimate_file, as_int=False):
    with open(estimate_file, "r+") as f:
        data = f.read()
        # creates a dictionary of (centerX, centerY) to radius
        circles = {}
        if (len(data) > 0):
            circle_data = data.split(";")
            for c in circle_data:
                circle = c.split(",")
                if (len(circle)) == 3 and not as_int:
                    circles[(float((circle[0])),float(circle[1]))] = float(circle[2])
                elif (len(circle)) == 3 and as_int:
                    circles[(int(float((circle[0]))),int(float(circle[1])))] = int(float(circle[2]))
    return circles

class BarnacleModel:
    def __init__(self, parameters):
        self.parameters = parameters

    def predict(self, imgs):
        return [self.predict_single(img) for img in imgs]

    def visualize_prediction(self, img):
        estimateFile = "output/est_" + img.split("/")[-1].split(".")[0] + ".txt"
        CircleViewer("", img, estimateFile).local_display()
        # try:
        #     CircleViewer("", img, estimateFile).local_display()
        # except Exception as e:
        #     print("No prediction found! Make sure the file path to the base image is provided.")
        
    def predict_single(self, img):
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]]) 
        sharpened_image = cv2.filter2D(gray, -1, kernel) 
        _, thresh = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, self.parameters["dp"], self.parameters["minDist"], 
                                   param1=self.parameters["param1"], param2=self.parameters["param2"], 
                                   minRadius=self.parameters["minRadius"], maxRadius=self.parameters["maxRadius"])        
        if circles is None:
            circles = [[]]

        filename = "output/est_" + img.split("/")[-1].split(".")[0] + ".txt"
        with open(filename, "w") as file:
            for c in circles[0]:
                file.write(f"{c[0]},{c[1]},{c[2]};")
        return len(circles[0])

    def score(self, imgs, counts):
        error = 0
        predictions = zip(self.predict(imgs), counts)
        for estimate, actual in predictions:
            error += abs((estimate - actual)/actual)
        return error/len(counts)

    def precise_score(self, images):
        self.predict(images)
        f1 = 0
        for img in images:
            f1 += self.precise_score_single(img)
        return f1/len(images)

    def precise_score_single(self, image):
        estimate_file = "output/est_img_" + image.split("img_")[1].split(".")[0] + ".txt"
        mask = "subsections/train/mask_" + image.split("img_")[1]
        
        estimate = extract_circles(estimate_file)
        true = unique_contours(mask)
        self.tp = set()
        fp = 0
        fn = 0

        found = set()

        for e in estimate.keys():
            bestMatch = None
            minDist = 99999999
            overlap = 0
            fullOverlap = 0
            fullOverlaps = []
            
            # estimate is accurate if:
            # a) overlaps with 1 or 2 actual circles not already matched with another prediction 
            # or b) if the inner 80% of the estimate contains only one actual circle center.
            for t in true.keys():
                if math.dist(e, t) < estimate[e] + true[t][0]:
                    overlap += 1
                    if math.dist(e, t) < minDist and not t in found:
                        bestMatch = t
                        minDist = math.dist(e, t)
                    if math.dist(e, t) < max(estimate[e],true[t][0]):
                        fullOverlaps.append(t)
                        fullOverlap += 1

            borderline = [x for x in fullOverlaps if math.dist(x,e) > estimate[e]*0.8]
            temp = fullOverlap - 1 == len(borderline) 
            if bestMatch != None and (overlap < 3 or fullOverlap == 1 or temp):
                found.add(bestMatch)
                self.tp.add((int(e[0]), int(e[1])))
            else:
                fp+=1
        tp = len(self.tp)
        fn = len(true) - len(found)
        precision = tp/(tp+fp) if tp + fp > 0 else 0
        recall = tp/(tp+fn) if tp + fn > 0 else 0
        return 2*precision*recall/(precision+recall) if precision + recall > 0 else 0

    def visualize_score(self, img):
        print(self.precise_score_single(img))
        estimateFile = "output/est_" + img.split("/")[-1].split(".")[0] + ".txt"
        mask = "subsections/train/mask_" + img.split("img_")[1]
        CircleViewer("", mask, estimateFile).local_display_score(self.tp)

class CircleViewer():

    def __init__(self, name, base_image, estimate_file):
        self.name = name
        self.image = cv2.imread(base_image)
        self.hovered = None
        self.circleStart = None
        self.lastDrawn = None
        self.circles = extract_circles(estimate_file, as_int=True)

    def local_display(self):
        newPaint = self.image.copy()
        for c in self.circles.keys():
            point = (c[0], c[1])
            radius = self.circles[point]
            circleColor = (0, 255, 0)
            thickness = 2
            cv2.circle(newPaint, point, radius, circleColor, thickness)
        plt.imshow(newPaint)

    def local_display_score(self, valid_circles):        
        newPaint = self.image.copy()
        # dict of center of contour to contour radius and area
        contour_dict = unique_contours(newPaint)
        for center in contour_dict.keys():
            cv2.circle(newPaint, center, int(contour_dict[center][0]), (0, 0, 0), 1)
        
        for c in self.circles.keys():
            point = (c[0], c[1])
            radius = self.circles[point]
            if point in valid_circles:
                circleColor = (0, 255, 0)
            else:
                circleColor = (255, 0, 0)
            thickness = 2
            cv2.circle(newPaint, point, radius, circleColor, thickness)
        plt.imshow(newPaint)

    def start(self):
        self.paint()
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.click_event)
        quit = False
        while (not quit):
            # waits for any key to be pressed
            key = cv2.waitKey(0)
            if (key == ord('q')):
                quit = True
            elif (key == ord('s')):
                self.save_edits()
                print("Saved to edited_output/est_" + self.name + ".txt")
            # elif (key == ord('n')):
            #     print("fjakl")
        cv2.destroyAllWindows()

    def save_edits(self):
        filename = "edited_output/est_" + self.name + ".txt"
        with open(filename, "w") as file:
            for c in self.circles.keys():
                x, y = c[0], c[1]
                r = self.circles[(x,y)]
                file.write(f"{x},{y},{r};")

    def paint(self):
        newPaint = self.image.copy()
        for c in self.circles.keys():
            point = (c[0], c[1])
            radius = self.circles[point]
            if point == self.hovered:
                circleColor = (0, 0, 255)
            else:            
                circleColor = (0, 255, 0)
            centerColor = (0, 0, 255)
            thickness = 2
            cv2.circle(newPaint, point, radius, circleColor, thickness)
            # if not drawing a circle
            if self.lastDrawn != point:
                cv2.circle(newPaint, point, 2, centerColor, thickness + 1)
        cv2.imshow(self.name, newPaint)

    def click_event(self, event, x, y, flags, params):
        point = (x,y)
        # if not drawing and  moving mouse
        if event == cv2.EVENT_MOUSEMOVE and self.circleStart == None:
            self.hovered = None
            # on mouse movement, check for hovered circle:
            keyset = set(self.circles.keys())
            # only one circle should be allowed to be within range
            for dx in range(-10,11):
                for dy in range(-10,11):
                    if (x + dx, y + dy) in keyset:
                        self.hovered = (x + dx, y + dy)
        # if drawing and moving mouse
        elif event == cv2.EVENT_MOUSEMOVE and self.circleStart != None:
            newCircleCenter = (int((self.circleStart[0] + x)/2), int((self.circleStart[1] + y)/2))
            newCircleRadius = int(math.sqrt((x - newCircleCenter[0])**2 + (y - newCircleCenter[1])**2))
            #deletes last drawn if it exists
            del self.circles[self.lastDrawn]
            self.lastDrawn = newCircleCenter
            self.circles[newCircleCenter] = newCircleRadius
        # if click
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.hovered != None:
                # delete hovered circle
                del self.circles[self.hovered]
                self.hovered = None
            else:
                self.circles[point] = 1
                self.circleStart = point
                self.lastDrawn = point
                #start a new circle
        # if mouse release
        elif event == cv2.EVENT_LBUTTONUP:
            if (self.lastDrawn != None and self.circles[self.lastDrawn] < 5):
                print("Circle too small to draw")
                del self.circles[self.lastDrawn]
            
            self.circleStart = None
            self.lastDrawn = None

        self.paint()