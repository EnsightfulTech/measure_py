import cv2
import math

def find_circles(image, min_points, max_offset, min_radius, max_radius,thrsh):
   
    result = []
    contours=preprocess(image,thrsh)
    
    centroids = []

    if len(contours) > 3000:
        return result

    for i in range(len(contours)):       
        
        if len(contours[i]) > min_points:
            center, radius = cv2.minEnclosingCircle(contours[i])
            offset = ComputeVariance(contours[i], center)
            
            if offset < max_offset and radius > min_radius and radius <= max_radius:
                area = cv2.contourArea(contours[i])
                min_area = math.pi * radius * radius * 2 / 3
                
                if area < min_area:
                    continue
                
                mu = cv2.moments(contours[i], False)
                mc = (mu["m10"] / mu["m00"], mu["m01"] / mu["m00"])
                rect = cv2.boundingRect(contours[i])
                ratio = rect[2] / rect[3]
                
                if abs(ratio - 1.0) < 0.28:
                    # p = (radius, ratio, area)  #修改该处为仅返回半径
                    p = radius
                    q = (mc[0], mc[1])
                    result.append(p)
                    centroids.append(q)
                    
                    cx=int(mc[0])
                    cy=int(mc[1])
                    
                    
                    cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 0, 255), 1)
                    cv2.circle(image,  (int(cx), int(cy)), 2, (0, 0, 255), -1)
                    cv2.putText(image, f"({int(cx)}, {int(cy)})", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    

    return result, centroids, image
