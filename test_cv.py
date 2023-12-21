
import cv2#导入包
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

from mpl_toolkits.mplot3d import Axes3D


def find_dirty_spot(img_test3,depth_map):
    LabImg = cv2.cvtColor(img_test3, cv2.COLOR_BGR2Lab)
    L,A,B = cv2.split(LabImg)
    blur = cv2.GaussianBlur(B, (3,3), 0)
    ret,thresh = cv2.threshold(blur,145,255,cv2.THRESH_BINARY)
    cv2.imshow("test3",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    k1 = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k1)

    cv2.imshow("test3",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sum_area=0
    points_array = []
    if len(contours)==0:
        sum_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        ellipse = cv2.fitEllipse(cnt)
        img_test3 = cv2.ellipse(img_test3,ellipse,(0,255,0),2)
        center, axes, angle = ellipse
        (h, k) = center
        (a, b) = axes

        # 计算焦点位置
        print("a:", a)
        print("b:", b)
        c = np.sqrt(abs((0.5*a)**2 - (0.5*b)**2))
        theta = np.radians(angle)  # 将角度转换为弧度
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        F1 = (int(h + c * sin_theta), int(k - c * cos_theta),depth_map[int(k - c * sin_theta), int(h + c * cos_theta)])
        F2 = (int(h - c * sin_theta), int(k + c * cos_theta),depth_map[int(k + c * sin_theta), int(h - c * cos_theta)])
        center=(int(h),int(k),depth_map[int(k),int(h) ])
   

        # print("F1:", F1)
        # print("F2:", F2)

        if area >= 200:
            sum_area=area+sum_area
            points_array.append(F1)
            points_array.append(center)
            points_array.append(F2)
            cv2.circle(img_test3, (F1[0], F1[1]), 2, (0, 0, 255), -1)
            cv2.circle(img_test3, (F2[0], F2[1]), 2, (0, 0, 255), -1)
            cv2.circle(img_test3, (center[0], center[1]), 2, (0, 0, 255), -1)        

            cv2.drawContours(img_test3,cnt,-1,(0,0,255),2)
        else:
            points_array.append(center) 
            cv2.circle(img_test3, (center[0], center[1]), 2, (0, 0, 255), -1)     
            cv2.drawContours(img_test3,cnt,-1,(0,0,255),2)
            




    print("\nFilled Points Array:")
    print(points_array) 
    cv2.imshow("test3",img_test3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dim1 = len(points_array)
    dim2 = 4
    dim3 = 4
    plane_points = [[[0 for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]


    posture=np.zeros((len(points_array), 3))
    temp_contours=contours
    # print(temp_contours[0][1][0])
    temp=0.0

    #number of contour,number of points in the contour,0,x/y
    for i in range(len(points_array)):

        # able1=0
        # able2=0
        # able3=0
        # able4=0
        
        # rows = len(contours[i])
        # cols = 3
        # temp_plane_poins = [[0 for _ in range(cols)] for _ in range(rows)]

        # for j in range(len(contours[i])):
            
        #     temp_contours[i][j][0][0]=contours[i][j][0][0]-points_array[i,0]
        #     temp_contours[i][j][0][1]=contours[i][j][0][1]-points_array[i,1]
        #     temp=(temp_contours[i][j][0][1]**2+temp_contours[i][j][0][0]**2)**0.5
        #     temp_plane_poins[j][0]=temp
        #     temp_plane_poins[j][1]=temp_contours[i][j][0][0]
        #     temp_plane_poins[j][2]=temp_contours[i][j][0][1]
        # temp_plane_poins.sort(key=lambda x: x[0])

        # for k in range(len(temp_plane_poins)):
        #     if(temp_plane_poins[k][1]>0 and temp_plane_poins[k][2]>0 and able1==0):
        #         plane_points[i][0][0]=temp_plane_poins[k][1]+points_array[i,0]
        #         plane_points[i][0][1]=temp_plane_poins[k][2]+points_array[i,1]
        #         depth_value = depth_map[int(plane_points[i][0][1]), int(plane_points[i][0][0])] 
        #         plane_points[i][0][2]=depth_value
        #         plane_points[i][0][3]=temp_plane_poins[k][0]
        #         able1=1
        #     if(temp_plane_poins[k][1]<0 and temp_plane_poins[k][2]>0 and able2==0):
        #         plane_points[i][1][0]=temp_plane_poins[k][1]+points_array[i,0]
        #         plane_points[i][1][1]=temp_plane_poins[k][2]+points_array[i,1]
        #         depth_value = depth_map[int(plane_points[i][1][1]), int(plane_points[i][1][0])]
        #         plane_points[i][1][2]=depth_value
        #         plane_points[i][1][3]=temp_plane_poins[k][0]
        #         able2=1
        #     if(temp_plane_poins[k][1]<0 and temp_plane_poins[k][2]<0 and able3==0):
        #         plane_points[i][2][0]=temp_plane_poins[k][1]+points_array[i,0]
        #         plane_points[i][2][1]=temp_plane_poins[k][2]+points_array[i,1]
        #         depth_value = depth_map[int(plane_points[i][2][1]), int(plane_points[i][2][0])] 
        #         plane_points[i][2][2]=depth_value
        #         plane_points[i][2][3]=temp_plane_poins[k][0]
        #         able3=1
        #     if(temp_plane_poins[k][1]>0 and temp_plane_poins[k][2]<0 and able4==0):
        #         plane_points[i][3][0]=temp_plane_poins[k][1]+points_array[i,0]
        #         plane_points[i][3][1]=temp_plane_poins[k][2]+points_array[i,1]
        #         depth_value = depth_map[int(plane_points[i][3][1]), int(plane_points[i][3][0])]
        #         plane_points[i][3][2]=depth_value
        #         plane_points[i][3][3]=temp_plane_poins[k][0]
        #         able4=1
        # plane_points[i].sort(key=lambda x: x[3],reverse=True)
        plane_points[i][0] = (points_array[i][0] + 10, points_array[i][1] + 10, depth_map[int(points_array[i][1] + 10), int(points_array[i][0]+10)] )
        plane_points[i][1] = (points_array[i][0] - 10, points_array[i][1] + 10, depth_map[int(points_array[i][1] - 10), int(points_array[i][0]+10)] )
        plane_points[i][2] = (points_array[i][0] , points_array[i][1] - 10, depth_map[int(points_array[i][1] - 8), int(points_array[i][0])] )

        
        v1= [plane_points[i][1][0]-plane_points[i][0][0],-plane_points[i][1][1]+plane_points[i][0][1],-float(plane_points[i][1][2])+float(plane_points[i][0][2])]

        v2= [plane_points[i][2][0]-plane_points[i][0][0],-plane_points[i][2][1]+plane_points[i][0][1],-float(plane_points[i][2][2]) + float(plane_points[i][0][2])]
  

        normal_vector = np.cross(v1, v2)
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
        posture[i]=normal_vector_normalized




    print("法线向量:", normal_vector_normalized)    
    

    # 创建一个三维图形对象
    # for i in range(len(posture)):
    #     start_point = np.array([0, 0, 0])
    #     end_point = np.array([posture[i,0], posture[i,1], posture[i,2]])

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # 绘制三维向量
    #     ax.quiver(start_point[0], start_point[1], start_point[2],
    #             end_point[0], end_point[1], end_point[2],
    #             color='b', arrow_length_ratio=0.1)
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1])
    #     ax.set_zlim([0, 1])

    #     # 设置坐标轴标签
    #     ax.set_xlabel('X轴')
    #     ax.set_ylabel('Y轴')
    #     ax.set_zlabel('Z轴')
    #     plt.show()
 
    # print(plane_points)
    return points_array,sum_area,posture

    
#print(temp_contours)

def find_real_coordinate(points_array,depth_map):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    # 创建对齐对象（深度对齐颜色）
    align = rs.align(rs.stream.color)
    while True:
        frames = pipeline.wait_for_frames()
            
            # 获取对齐帧集
        aligned_frames = align.process(frames)
            
            # 获取对齐后的深度帧和彩色帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

            # 获取颜色帧内参
        color_profile = color_frame.get_profile()
        cvsprofile = rs.video_stream_profile(color_profile)
        color_intrin = cvsprofile.get_intrinsics()
        color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]
        ppx = color_intrin_part[0]
        ppy = color_intrin_part[1]
        fx = color_intrin_part[2]
        fy = color_intrin_part[3]
        for i in range(len(points_array)):


            target_xy_pixel = [points_array[i,0], points_array[i,1]]
            target_depth = depth_map[int(points_array[i,1]), int(points_array[i,0])]  
            target_xy_true = [(target_xy_pixel[0] - ppx) * target_depth / fx,
                                (target_xy_pixel[1] - ppy) * target_depth / fy]
            points_array[i,0]=target_xy_true[0]
            points_array[i,1]=target_xy_true[1]        
            print(target_xy_pixel[0],target_xy_pixel[1],target_xy_true[0]/10,-target_xy_true[1]/10,target_depth)
        break 
    camera_pos=[0,50,70]
    points_array_ture=points_array
    for i in range(len(points_array)):
        points_array_ture[i,0]=points_array[i,0]+camera_pos[0]
        points_array_ture[i,1]=points_array[i,1]+camera_pos[1]
        points_array_ture[i,2]=points_array[i,2]+camera_pos[2]
    return points_array_ture

before = cv2.imread("/home/pthahnil/Desktop/realsense/before.jpeg")
before = cv2.resize(before, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_AREA)
after = cv2.imread("/home/pthahnil/Desktop/realsense/after.jpeg")
after = cv2.resize(after, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_AREA)
depth_map = cv2.imread('/home/pthahnil/Desktop/realsense/1_depth.png', cv2.IMREAD_UNCHANGED)

area_before=find_dirty_spot(before,depth_map)[1]
area_after=find_dirty_spot(after,depth_map)[1]
print((area_before-area_after)*100/area_before)
# img = cv2.imread("/home/pthahnil/Desktop/realsense/1.jpg")
# img = cv2.resize(img, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_AREA)
# find_dirty_spot(img,depth_map)

