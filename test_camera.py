import pyrealsense2 as rs
import numpy as np
import cv2
import h5py

class Camera():
    def __init__(self, width=640, height=480, fps=30, ColorAndDepth = False, video_folder = None):
        
        self.width = width
        self.height = height
        self.fps = fps
        self.isrecording = False
        self.ColorAndDepth = ColorAndDepth
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.video_folder = video_folder
        
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  fps)
        self.video_folder = video_folder
        
        # self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, fps)
        # self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, fps)
        self.pipeline.start(self.config)    

    def getframe(self):
        # Get the frames from external realsense camera
        frame = self.pipeline.wait_for_frames()
        return frame

    def getframedata(self, frame):
        #Retrieves the Depth and color values from the frames and convert them to a np array
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image
    
    def alignresolution(self, frame):
        align_to = rs.stream.color                 
        align = rs.align(align_to)
        aligned_frame = align.process(frame)

        aligned_depth_frame = aligned_frame.get_depth_frame()      
        aligned_color_frame   = aligned_frame.get_color_frame()

        return aligned_frame, aligned_depth_frame, aligned_color_frame 
    
    def Datatreatment(self):
        
        # Get the frames from external realsense camera
        frame = self.getframe()
        
        #Setting colours to different levels of depth
        colorizer = rs.colorizer()

        #Retrieves the Depth and color values from the frames and convert them to a np array
        depth_image, color_image = self.getframedata(frame)
        

        #Align the resolution of the depth and color of the image. 
        aligned_frame, aligned_depth_frame, aligned_color_frame = self.alignresolution(frame)

        #Setting colors to the aligned depth frame
        colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        #Retrieve the depth and color values from the aligned frame and convert it to an array
        aligned_depth_image, aligned_color_image = self.getframedata(aligned_frame)
        

        return color_image, aligned_depth_image, aligned_color_image, colorizer_depth
    
    def release(self):
        self.pipeline.stop()


    def stream(self):
        try:
            print("press q to quit and t to toggle")
            while True:
                _, _, aligned_color_image, colorizer_depth = self.Datatreatment()
                
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                #Portray the colorizer depth
                if self.ColorAndDepth:
                    cv2.imshow('RealSense', colorizer_depth)
                else:
                    cv2.imshow('RealSense', aligned_color_image)
                
                key = cv2.waitKey(1)
                if (key & 0xFF == ord('q') or key == 27):
                    cv2.destroyAllWindows()
                    break
                if key & 0xFF == ord('t') :
                    cv2.destroyAllWindows()
                    if self.ColorAndDepth == True:
                        self.ColorAndDepth = False
                    else:
                        self.ColorAndDepth = True
        
        finally:
            self.release()
    
    def storingfilepath(self):
        
        video_path = self.video_folder + "/targetvideo_rgb.mp4"
        video_depthcolor_path = self.video_folder + "/targetvideo_depthcolor.mp4"
        video_depth16_path = self.video_folder + "/targetvideo_depth.h5"
        photo_path = self.video_folder + "/photo.jpg"
        photo_depth_path = self.video_folder + "/photo_depth.png"

        return video_path, video_depthcolor_path, video_depth16_path, photo_path, photo_depth_path

    def initialise_recording_function(self):
        video_path, video_depthcolor_path, video_depth16_path, _, _= self.storingfilepath()
        
        mp4 = cv2.VideoWriter_fourcc(*'mp4v') 

        #Initialising different recording functions. normal rgb, depth and color+depth
        wr = cv2.VideoWriter(video_path, mp4, self.fps, (self.width, self.height), isColor=True)
        wr_colordepth = cv2.VideoWriter(video_depthcolor_path, mp4, self.fps, (self.width, self.height), isColor=True)
        wr_depth = h5py.File(video_depth16_path, 'w')

        return wr, wr_colordepth, wr_depth

    def recording(self):
        if self.video_folder is None:
            raise Exception("Please input in your video folder /Usr.../folder")
        try:
            print("Press s to record, t to toggle and q to quit")

            #Retrieve all the file Path
            video_path, _, _, photo_path, photo_depth_path = self.storingfilepath()

            idx = 0
            
            while True:
                #Retrieve Data
                color_image, aligned_depth_image, aligned_color_image, colorizer_depth = self.Datatreatment()

                #Initialise Window
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

                #Shift between Color&depth and RGB
                if self.ColorAndDepth:
                    cv2.imshow('RealSense', colorizer_depth)
                else:
                    cv2.imshow('RealSense', color_image)

                key = cv2.waitKey(1)
 
                if key & 0xFF == ord('s') and self.isrecording is False :
                    self.isrecording = True
                    
                    #Initialising different recording functions.
                    wr, wr_colordepth, wr_depth = self.initialise_recording_function()

                    print('...recording...')
                #Allow to toggle between Colr&Depth and RGB
                if key & 0xFF == ord('t'):
                    if self.ColorAndDepth == True:
                        self.ColorAndDepth = False
                    else:
                        self.ColorAndDepth = True
                if self.isrecording:
                    
                    #Writing the color and colorizer depth to the different recording system, wr and wr_colordepth
                    wr.write(color_image)                         
                    wr_colordepth.write(colorizer_depth)         
                    
                    #Encodes the depth array into a photo
                    depth16_image = cv2.imencode('.png', aligned_depth_image)[1]
                    
                    #Save the depth16 image to a unique name inside the h5py file
                    depth_map_name = str(idx).zfill(5) + '_depth.png'
                            
                    wr_depth[depth_map_name] = depth16_image          
                    
                    idx += 1
                    
                if (key & 0xFF == ord('q') or key == 27) and self.isrecording:

                    cv2.imwrite(photo_path, color_image)
                    cv2.imwrite(photo_depth_path, aligned_depth_image)
                    cv2.destroyAllWindows()
                    print('...quit...')
                    
                    break
            
        finally:
            print(f"The video is saved at {video_path}")
            self.release()


if __name__ == "__main__":

    c = Camera(video_folder="/Users/joshua/vscode/hivebotics/robot_computor_vision/realsense")
    
    c.recording()

   