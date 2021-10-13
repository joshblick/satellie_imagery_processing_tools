import numpy as np
import cv2
import math
import copy
import os
from operator import itemgetter, attrgetter
from matplotlib.pyplot import subplots

#TODO: capital letters for classes

class detection:
    """
    """
    def __init__(self, box_array, score, class_no, detection_number, parent_snippet_y_min, parent_snippet_x_min, parent_snippet_size):
        #set box limits 
        ymin_ratio_np, xmin_ratio_np, ymax_ratio_np, xmax_ratio_np = box_array
        self.ymin_ratio, self.xmin_ratio, self.ymax_ratio, self.xmax_ratio = float(ymin_ratio_np), float(xmin_ratio_np), float(ymax_ratio_np), float(xmax_ratio_np)
        
        #set the score and the class number
        self.score = float(score.numpy()) #TODO: do something clean
        self.class_no = float(class_no.numpy()) #TODO: do something clean
        # reverse score for sorting 
        self.detection_score_sort = 1.0-score
        # convert the local coordinates within the snippet (ratios) to clobal image coordinates
        
        self.y_min, self.x_min, self.y_max, self.x_max = self.find_global_coordinates(parent_snippet_y_min = parent_snippet_y_min, 
                                                                                        parent_snippet_x_min = parent_snippet_x_min, 
                                                                                        parent_snippet_size = parent_snippet_size
                                                                                        )
        
        # set the status of the detection to "active" as opposed to deleted when there is a matching item
        
        self.active = True
        
        #name is parent name plus the snippet_number
        
        #get the centroid of the bounidng box
        self.centroidx = (self.x_max + self.x_min)/2.0
        self.centroidy = (self.y_max + self.y_min)/2.0
        
        # get distance across the detection box
        
        self.quasi_radius = self.find_quasi_radius()
        
        # add the size of the box        
        self.size = (self.y_max - self.y_min)*(self.x_max - self.x_min)
        
        # set the id which references the objects position in the image list and which will be overwritten by the image method
        self.id = None
        
        self.possible_overlaps = []
        
        self.overlaps = []
    
    def identify_possible_overlaps(self, all_detections_in_image, num_quasi_radii):
        """Given a list of detection objects at the image level, this function identify those for which it is 
        worthwhile to check the intersection

        The rule is basically that if they are worth being checked the sum of the quasi-radii of the two 
        boxes will be less than the distance between them.
        """
        #TODO: come up with a more robust methodolgy than the differnce in x_mins vs radii shit
        possible_overlaps = [detection for detection in all_detections_in_image if detection.id!=self.id]
        
        possible_overlaps = [detection for detection in possible_overlaps if abs(detection.x_min-self.x_min)<=(self.quasi_radius+detection.quasi_radius)]
        
        self.possible_overlaps = [detection.id for detection in possible_overlaps if abs(detection.y_min-self.y_min)<=(self.quasi_radius+detection.quasi_radius)]
        
    def identify_overlaps(self, all_detections_in_image, intersection_over_union_cutoff):
        """
        Given the list of ids (that is ids in the original image which represent the location in the image's detections attributie (list)
        and the provided iou cutoff and preference for cars, we perform a 
        
        """
        
        for possible_overlap_id in self.possible_overlaps:
            #compare intersection over union (IOU) with cutoff and add to 
            other_detection = all_detections_in_image[possible_overlap_id]
            iou = self.calculate_IOU(other_detection)
            if iou>=intersection_over_union_cutoff:
                self.overlaps.append(other_detection.id)
                
    
    def find_quasi_radius(self):
        a_squared = (self.y_max - self.y_min)**2
        b_squared = (self.x_max - self.x_min)**2
        
        return math.sqrt(a_squared + b_squared)/2
        
    def find_global_coordinates(self, parent_snippet_y_min, parent_snippet_x_min, parent_snippet_size):
        """Function which finds the location of an object in the 
           scope of the original image, based on the relative coordinates
           provided in the 
        
        """
        x_offset = parent_snippet_x_min
        y_offset = parent_snippet_y_min
        
        y_min = y_offset + (self.ymin_ratio*parent_snippet_size)
        y_max = y_offset + (self.ymax_ratio*parent_snippet_size)
        x_min = x_offset + (self.xmin_ratio*parent_snippet_size)
        x_max = x_offset + (self.xmax_ratio*parent_snippet_size)
        
        return y_min, x_min, y_max, x_max
    
    def set_id(self, new_id):
        """
        Set an id corresponding to the location of the detection object in the 
        list of detection objects at the image level
        """
        self.id = new_id
    
    def calculate_IOU(self, other):
        """
        Calculates the IOU
        """
        
        intersection_y_max = min(self.y_max, other.y_max)
        intersection_x_max = min(self.x_max, other.x_max)
        intersection_y_min = max(self.y_min, other.y_min)
        intersection_x_min = max(self.x_min, other.x_min)
        
        intersection_size = (intersection_y_max - intersection_y_min)*(intersection_x_max - intersection_x_min)
        
        return intersection_size/(self.size + other.size - intersection_size)
    
    def print_bounds(self):
        print("y_min: {0}, x_min: {1}, y_max: {2}, x_max: {3}".format(round(self.y_min,1),
                                                                        round(self.x_min,1),
                                                                        round(self.y_max,1),
                                                                        round(self.x_max,1)
                                                                        ))

class snippet:
    """
    Attributes:
     - image_as_nparray
     - y_min
     - y_max
     - x_min
     - x_max
     - parent_image_name
     - <outdated>detections</outdated>

     
    Methods:
     - __init__(self, image_as_nparray, snippet_size, upper_limit, lower_limit, parent_image_name
    """
    def __init__(self, snippet_size, x_min, y_min, image_as_nparray, image_xmax,
                image_ymax, parent_image_name, is_offset):
        """
        """
        self.x_min = x_min
        self.y_min = y_min
        self.size = snippet_size
        self.parent_image_name = parent_image_name
        self.is_offset = is_offset
        
        self.x_max, self.y_max, self.padding_x, self.padding_y = self.get_appropriate_bounds(x_min = x_min, 
                                                                                            y_min = y_min, 
                                                                                            snippet_size = snippet_size, 
                                                                                            image_xmax = image_xmax, 
                                                                                            image_ymax = image_ymax
                                                                                            )
        
        self.name = "{0}_{1}_{2}_{3}".format(self.x_min, self.x_max, self.y_min, self.y_max) 
        if (self.padding_x==0) & (self.padding_y==0):
            self.image_as_nparray = image_as_nparray[self.y_min: self.y_max, self.x_min: self.x_max]
        else:
            self.image_as_nparray = cv2.copyMakeBorder(image_as_nparray[self.y_min: self.y_max, self.x_min: self.x_max], 
                                                        0, 
                                                        self.padding_y, 
                                                        0, 
                                                        self.padding_x, 
                                                        cv2.BORDER_CONSTANT
                                                        )
    
    def return_detected_objects(self, detection_function, score_cutoff):
        """
        GENERATEOR
        Detects the objects in the image snipet using the provided TF detection function
        and then converts each one to a detection object (through the init method of the 
        detection class). 
        ## update: the detections are no longer added to the snippet, they are just passed to image
        ####<Outdated>
        ##Each object detected (above the given confidence level cutoff) 
        #is added to the detections list attribute.
        ####</Outdated>
        
        In order to allow the comparison of detections across different snippets, in addition
        to writing the detecion objects to a list attribute in the sniipet, it also acts as a
        generator which yields each detectio object
        
        
        """
        
        #TODO: take advantage of the batch processing tensor function instead of just doing this:
        #reading just one image
        test_tensor = tf.convert_to_tensor(self.image_as_nparray)
        input_tensor = test_tensor[tf.newaxis, ...]
        
        #apply the model to run inference
        detections = detection_function(input_tensor)
        
        #for each potential detection in the result we generate a detection object named i
        i=0
        for box_limits, score, class_no in zip(detections["detection_boxes"][0], detections["detection_scores"][0], detections["detection_classes"][0]):
            if score>score_cutoff:
                detected_object = detection(box_array = box_limits,
                                            score = score, 
                                            class_no = class_no, 
                                            detection_number = i,
                                            parent_snippet_y_min = self.y_min, 
                                            parent_snippet_x_min = self.x_min, 
                                            parent_snippet_size = self.size
                                            )
                i+=1
                yield detected_object
    
    def get_appropriate_bounds(self, x_min, y_min, snippet_size, image_xmax, image_ymax):
        """
        need to identify the bounds of the wider image to pull and then padd 
        them with black as necessary to make all images the correct size
        """
        if (x_min + snippet_size > image_xmax) & (y_min + snippet_size > image_xmax):
            x_max, y_max = image_xmax, image_ymax
            padding_x = (x_min + snippet_size) - image_xmax
            padding_y = (y_min + snippet_size) - image_ymax
            
        elif (x_min + snippet_size > image_xmax):
            x_max, y_max = image_xmax, y_min + snippet_size
            padding_x = (x_min + snippet_size) - image_xmax
            padding_y = 0
            
        elif (y_min + snippet_size > image_xmax):
            x_max, y_max = x_min + snippet_size, image_ymax
            padding_x = 0
            padding_y = (y_min + snippet_size) - image_ymax
            
        else:
            x_max, y_max = x_min + snippet_size, y_min + snippet_size
            padding_x, padding_y = 0,0
        
        return x_max, y_max, padding_x, padding_y

class image:
    """
    Attributes:
    - image_path
    - image_name
    - store_number
    - month
    - year
    - image_as_nparray

    """ 
    def __init__(self, image_path, image_name="from_filename"):
        if image_name!="from_filename":
            self.image_name = image_name
        else:
            self.image_name = image_path.split("\\")[-1].split(".")[0]
        self.image_path = image_path
        
        # obtain store details and image date from the filename
        details = self.image_name.split("_")
        self.store_number = details[0]
        self.month = details[1]
        self.year = details[2]
        
        # pull in image as an nparray, get dimensions
        self.image_as_nparray = cv2.imread(image_path)
        self.y_max = self.image_as_nparray.shape[0]
        self.x_max = self.image_as_nparray.shape[1]
        
        # add empty lists for both image snippet objects and detections which will be added to by methods
        self.snippets = []
        self.detections = []
    
    def cut_image(self, output_size, xy_inialisation, x_max, y_max, is_offset):
        """
        """
        rows = 0
        images = 0
        #check inputs
        if (type(output_size) != int) | (type(x_max) != int) | (type(y_max) != int):
            print("{0}, {1}, {2} must be int".format("output_size" if (type(output_size) != int) else "",
                                                    "x_max" if (type(x_max) != int) else "",
                                                    "y_max" if (type(y_max) != int) else ""))
                 #we will scan across the image left to right then down
        y = xy_inialisation
        while (y_max-y)>=output_size:
            rows += 1
            x = xy_inialisation
            
            while (x_max-x)>=output_size:
                images += 1
                self.snippets.append(snippet(snippet_size = output_size, 
                                            x_min = x, 
                                            y_min = y, 
                                            image_as_nparray = self.image_as_nparray, 
                                            image_xmax = self.x_max,
                                            image_ymax = self.y_max, 
                                            parent_image_name = self.image_name,  
                                            is_offset = is_offset))
                x += output_size
            else:
                images += 1
                self.snippets.append(snippet(snippet_size = output_size, 
                                            x_min = x, 
                                            y_min = y, 
                                            image_as_nparray = self.image_as_nparray, 
                                            image_xmax = self.x_max,
                                            image_ymax = self.y_max, 
                                            parent_image_name = self.image_name,  
                                            is_offset = is_offset))
                x += output_size
            y += output_size
        else:
            rows += 1
            x = xy_inialisation
            
            while (x_max-x)>=output_size:
                images += 1
                self.snippets.append(snippet(snippet_size = output_size, 
                                            x_min = x, 
                                            y_min = y, 
                                            image_as_nparray = self.image_as_nparray, 
                                            image_xmax = self.x_max,
                                            image_ymax = self.y_max, 
                                            parent_image_name = self.image_name,  
                                            is_offset = is_offset))
                x += output_size
            else:
                images += 1
                self.snippets.append(snippet(snippet_size = output_size, 
                                            x_min = x, 
                                            y_min = y, 
                                            image_as_nparray = self.image_as_nparray, 
                                            image_xmax = self.x_max,
                                            image_ymax = self.y_max, 
                                            parent_image_name = self.image_name,  
                                            is_offset = is_offset))
                x += output_size
            y += output_size
        return int(rows), int(images/rows)
    
    def split_into_snippets(self, output_size, with_offsets):
        """
        output_size:    (int) the required height and width of the image
        with_offsets:   (bool) whether or not offsetting snippets i.e., 
                        images starting output_size/2 from the edge and 
                        finishing in the same place on the other side, top, bottom
        
        Returns:        list of snippet objects added to the snippets list attribute
        """
        #offset adjustment in case of splitting with offsets 
        offset_adjustment = (output_size)/2
        
        # add the cut images to the snippets attribute
        self.rows, self.columns = self.cut_image(output_size, 
                                                 0, 
                                                 self.x_max, 
                                                 self.y_max,
                                                 is_offset = False) #for the regular
        if with_offsets:#for the offsets
            self.offset_rows, self.offset_columns = self.cut_image(output_size, 
                                                                    int(offset_adjustment), 
                                                                    int(self.x_max - offset_adjustment), 
                                                                    int(self.y_max - offset_adjustment), #TODO: don't just be lazy and cast, work out why they're ints atm
                                                                    is_offset = True)
    def plot_image_snippets(self, fig_size):
        """
        """#TODO: split into two or give option to choose which lone to plot
        fig,axes = plt.subplots(nrows = self.rows, ncols = self.columns, figsize=(fig_size,fig_size))
        ##label as numbers for each image
        image_no=0
        for i in range(self.rows):
            for j in range(self.columns):
                axes[i,j].imshow(self.snippets[image_no].image_as_nparray)
                axes[i,j].axis('off')
                image_no += 1
        plt.show()
        
        fig,axes = plt.subplots(nrows = self.offset_rows, ncols = self.offset_columns, figsize=(fig_size,fig_size))
        image_no = self.rows*self.columns
        for i in range(self.offset_rows):
            for j in range(self.offset_columns):
                axes[i,j].imshow(self.snippets[image_no].image_as_nparray)
                axes[i,j].axis('off')
                image_no += 1
        plt.show()
    
    def detect_objects(self, detection_function, score_cuttoff):
        """
        Detects all objects in the snippets within this image
        """
        # set a detection_counter so that each detection can be referenced by its position in the image.detections list
        i=0
        for snippet in self.snippets:
            for detection in snippet.return_detected_objects(detection_function = detection_function, score_cutoff = score_cuttoff):
                self.detections.append(detection)
                detection.set_id(i)
                i+=1
    
    def remove_overlapping_objects(self, num_quasi_radii, intersection_over_union_cutoff, preference_cars = True):
        """
        Identifies detection objects with overlapping bounding boxes and then Sets the active attribute 
        of the detection(s) with lower score 
        Preferences detectuibs which are inferred to be cars regardless of score by default
        """
        
        #TODO: this is super messy, should be in sub functions, look into it.
        #it is important that the process of setting active = False is undertaken from the objects
        #with the highest scores first. However if I sort te objects it;s going to fuck up the system of refering
        #the objects based on their position in the detection lists. So instad we get a list of the ids in order
        # if there is a preference for cars this sorting needs to be first on the clas id (it's 1 for cars)
        if preference_cars:
            order_to_process_detections = [detection.id for detection in sorted(self.detections, key=attrgetter("class_no", "detection_score_sort"))]
        else:
            order_to_process_detections = [detection.id for detection in sorted(self.detections, key=attrgetter("class_no"))]
        # then we perform the process of deactivating overlapping boxes
        
        for detection_id in order_to_process_detections:
            #take the detection from the list of detections
            detection = self.detections[detection_id]
            #if it's active (i.e. those with higher scores haven't deactived it)
            if detection.active:
                #idenify possible overlaps
                detection.identify_possible_overlaps(all_detections_in_image = self.detections, num_quasi_radii = num_quasi_radii)
                #identify overlaps based on the 
                detection.identify_overlaps(all_detections_in_image = self.detections, intersection_over_union_cutoff = intersection_over_union_cutoff)
                
                #set these detections to inactive
                for overlapping_id in detection.overlaps:
                    self.detections[overlapping_id].active = False
    
    def plot_active_detections_cv2(self, circle_radius_ratio, circle_brg_colour_tuple, size):
        # plot the full image with small cicles over the active detection objects
        self.copy_for_plotting = copy.deepcopy(self.image_as_nparray)
        fig, ax = subplots(figsize=size)
        #plot the circles
        for detection in self.detections:
            if detection.active:
                cv2.circle(img = self.copy_for_plotting, center = (int(detection.centroidx), 
                                                                    int(detection.centroidy)), 
                                                                    radius = int(detection.quasi_radius*circle_radius_ratio), 
                                                                    color = circle_brg_colour_tuple,
                                                                    thickness=-1)
        ax.imshow(self.copy_for_plotting, interpolation='nearest')
        
    def number_of_cars(self):
        return len([item for item in self.detections if item.active])
print("Classes written correctly")