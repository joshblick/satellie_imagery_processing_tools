import cv2

class snippet:
    """
    Attributes:
     - image_as_nparray
     - y_min
     - y_max
     - x_min
     - x_max
     - parent_image_name
     - detections
     
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
        
        
        self.x_max, self.y_max, self.padding_x, self.padding_y = get_appropriate_bounds(x_min = x_min, 
                                                                                        y_min = y_min, 
                                                                                        snippet_size = snippet_size, 
                                                                                        image_xmax = image_xmax, 
                                                                                        image_ymax = image_ymax
                                                                                        )
        
        self.name = "" #TODO: come up with some sort of naming convention like [y_min]_[x_min]_[y_max]_[x_max]
        if (padding_x==0) & (padding_y==0):
            self.image_as_nparray = image_as_nparray[y_min: y_max, x_min: x_max])
        else:
            self.image_as_nparray = cv2.copyMakeBorder(image_as_nparray[y_min: y_max, x_min: x_max], 
                                                        0, 
                                                        self.padding_y, 
                                                        0, 
                                                        self.padding_x, 
                                                        cv2.BORDER_CONSTANT
                                                        )
        
    def self.get_appropriate_bounds(x_min, y_min, snippet_size, image_xmax, image_ymax):
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
    
    def cut_image(self, output_size, xy_inialisation, x_max, y_max):
        """
        """
        rows = 0
        images = 0
        
        #check inputs
        if (type(output_size) != int) | (type(x_max) != int) | (type(y_max) != int):
            print("{0}, {1}, {2} must be int".format("output_size" if (type(output_size) != int) else "",
                                                    "x_max" if (type(x_max) != int) else "",
                                                    "y_max" if (type(y_max) != int) else "")
                 )
        #we will scan across the image left to right then down
        y = xy_inialisation
        while (y_max-y)>=output_size:
            #if there is the full pixels needed 
            #we initiate the counter in the x direction
            rows += 1
            x = xy_inialisation
            
            while (x_max-x)>=output_size:
                images += 1
                #add the cut piece of the image to the list
                self.snippets.append(self.image_as_nparray[y:y+output_size, x:x+output_size])
                x += output_size
            else:
                images += 1
                
                ##### Start of init function in snippet object ####
                
                #cut the crop image
                crop_img=self.image_as_nparray[y:y+output_size, x:x_max]
                #create the border around the remaining part
                bordered_im = cv2.copyMakeBorder(crop_img, 0, output_size-crop_img.shape[0],\
                                             0 ,output_size-crop_img.shape[1], cv2.BORDER_CONSTANT)
                
                #### End of init function in snippet object ####
                
                self.snippets.append(bordered_im)
                x += output_size
            y += output_size
        else:
            rows += 1
            #if we need to take a slice less than the 500 
            #pixels in the y direction and add padding
            #we initiate the counter in the x direction
            x = xy_inialisation
            
            while (x_max-x)>=output_size:
                images += 1
                
                ##### Start of init function in snippet object ####
                
                #cut the crop image
                crop_img=self.image_as_nparray[y:y_max, x:x+output_size]
                #create the border around the remaining part
                bordered_im = cv2.copyMakeBorder(crop_img, 0, output_size-crop_img.shape[0],\
                                                 0,output_size-crop_img.shape[1], cv2.BORDER_CONSTANT)
                
                #### End of init function in snippet object ####
                
                self.snippets.append(bordered_im)
                x += output_size
            else:
                images += 1
                
                ##### Start of init function in snippet object ####
                
                #cut the crop image
                crop_img=self.image_as_nparray[y:y_max, x:x_max]
                #create the border around the remaining part
                bordered_im = cv2.copyMakeBorder(crop_img, 0, output_size-crop_img.shape[0],\
                                                 0,output_size-crop_img.shape[1], cv2.BORDER_CONSTANT)
                
                #### End of init function in snippet object ####
                
                self.snippets.append(bordered_im)
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
                                                 self.y_max
                                                ) #for the regular
        if with_offsets:#for the offsets
            self.offset_rows, self.offset_columns = self.cut_image(output_size, 
                                                                    int(offset_adjustment), 
                                                                    int(self.x_max - offset_adjustment), 
                                                                    int(self.y_max - offset_adjustment) #TODO: don't just be lazy and cast, work out why they're ints atm
                                                                    )
    def plot_image_snippets(self, fig_size):
        """
        """#TODO: split into two or give option to choose which one to plot
        fig,axes = plt.subplots(nrows = self.rows, ncols = self.columns, figsize=(fig_size,fig_size))
        ##label as numbers for each image
        image_no=0
        for i in range(self.rows):
            for j in range(self.columns):
                axes[i,j].imshow(self.snippets[image_no])
                axes[i,j].axis('off')
                image_no += 1
        plt.show()
        
        fig,axes = plt.subplots(nrows = self.offset_rows, ncols = self.offset_columns, figsize=(fig_size,fig_size))
        image_no = self.rows*self.columns
        for i in range(self.offset_rows):
            for j in range(self.offset_columns):
                axes[i,j].imshow(self.snippets[image_no])
                axes[i,j].axis('off')
                image_no += 1
        plt.show()
        