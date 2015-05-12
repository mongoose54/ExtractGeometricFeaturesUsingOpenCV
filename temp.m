#pragma mark -
#pragma mark - Main function where feature extraction happens
//Find Big Round Region in the center
//Use Binarization with Otsu algorithm
//Find single and biggest region


-(void) extractGeometricFeaturesFromRegionOfInterest:(UIImage *) image{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    
    //Some auxilary matrices to store intermediate outputs
    cv::Mat inputImage,copyInputImage,tempThresholdFrame,otsu,markerMask,thresholded,thresholded_All,HSV, HSV_All;
    
    //Save initial input image into OpenCV object
    inputImage = [image CVMat];
    
    //Make a copy of the original input image into OpenCV object
    copyInputImage = [image CVMat];
    
    
    //Perform binarization using Otsu method
    cv::cvtColor(inputImage, tempThresholdFrame, cv::COLOR_RGB2GRAY);
    
    //Toggle this flag if you want to change manual/automatic thresholding
    BOOL manual=FALSE;
    if (manual==TRUE)
        cv::threshold(tempThresholdFrame,otsu,1,255,CV_THRESH_BINARY);
        else
            cv::threshold(tempThresholdFrame,otsu,100,255,CV_THRESH_OTSU);
            
            
            cv::Mat temporary(tempThresholdFrame.rows,tempThresholdFrame.cols,CV_8UC1);
            
            //Find Contours on Binary Image
            cv::vector<cv::Mat> contours;
    cv::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(otsu, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    //Create a temporary empty image to store contours of ROI for display purposes
    cv::Mat tempROIonInputImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC4);
    
    //Perform the contour drawing
    BOOL successROI=FALSE;
    double totalAreaOfMask=0;
    double maskPerimeter=0;
    cv::Scalar color = CV_RGB(  255,  255,  255 );
    for (int i=0; i<contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area>totalAreaOfMask) {
            cv::drawContours(tempROIonInputImage, contours, i, color, CV_FILLED, 8, hierarchy);
            cv::drawContours(temporary, contours, i, color, CV_FILLED, 8, hierarchy);
            maskPerimeter =cv::arcLength(contours[i], YES);
            successROI=TRUE;
            totalAreaOfMask=area;
        }
    }
    
    
    //Check if binary image has at least one big ROI
    if (successROI==TRUE) {
        
        //In the meantime update progrss bar to display stage we are in
        [self performSelectorOnMainThread:@selector(updateProgress:) withObject:[NSNumber numberWithInt:2] waitUntilDone:YES];
        
        //Find rectangle of ROI
        cv::Mat ROIFrame=[self getROIFrom:inputImage withMask:tempROIonInputImage];
        
        //Display ROI
        UIImage *ROIFrameImage = [UIImage imageWithCVMat:ROIFrame];
        [self displayImage:ROIFrameImage];
        
        //Set ROI image to HSV color space
        cv::cvtColor(ROIFrame, HSV, cv::COLOR_RGB2HSV);
        
        //Set some experimental threshold values
        cv::Scalar hsv_min = cvScalar(0, 50, 0, 0);
        cv::Scalar hsv_max = cvScalar(255, 200, 256, 0);
        cv::Scalar hsv_min2 = cvScalar(170, 50, 170, 0);
        cv::Scalar hsv_max2 = cvScalar(256, 180, 256, 0);
        
        //Binarize HSV image
        cv::inRange(HSV, hsv_min, hsv_max, thresholded);
        
        //Apply morphological operation
        cv::morphologyEx(thresholded,thresholded,1,cv::Mat::ones(10, 10, CV_8UC1));
        
        //Find Contours in the binarized image
        cv::findContours(thresholded, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        
        //NSLog(@"Moments: %f , %f",cv::moments(temporary).m10/cv::moments(temporary).m00,cv::moments(temporary).m01/cv::moments(temporary).m00);
        
        //Draw contour around one region: melanoma (hopefully!)
        cv::Mat maskROI = cv::Mat::zeros(ROIFrame.rows, ROIFrame.cols, CV_8UC4);
        cv::Scalar red = CV_RGB(0,0,250);
        
        //These are the main geometric features
        double area=0;
        double perimeter=0;
        double circularity=0;
        double effectiveDiameter=0;
        double compactness=0;
        double shapeIndex=0;
        
        cv::Scalar blackColor = CV_RGB(0,  0,  0 );
        
        double contourArea=0;
        int contourID=0;
        for (int i=0; i<contours.size(); i++)
            if ((cv::contourArea(contours[i])>VideoQuality*VideoQuality*50000.00)&&(cv::contourArea(contours[i])>contourArea)) {
                contourArea = cv::contourArea(contours[i]);
                contourID=i;
            }
        
        cv::drawContours(maskROI, contours, contourID, color, CV_FILLED, 8);
        cv::drawContours(ROIFrame, contours, contourID, blackColor, 5);
        
        
        //Calculate the geometri characteristics
        area = abs(cv::contourArea(contours[contourID]));  //Physical size of  ROI
        perimeter = cv::arcLength(contours[contourID], YES); //Perimeter of contour for ROI
        compactness = cv::arcLength(contours[contourID], YES) /(2*1.772*area); //Compactness of ROI
        circularity = (2*area*1.772)/cv::arcLength(contours[contourID], YES);  //Circularity of ROI
        effectiveDiameter = 2* sqrt(area/3.14159);  //Effective diameter of ROI
        shapeIndex = cv::arcLength(contours[contourID], YES)/(2*sqrt(3.14*area));  //Shape index of ROI
        
        NSLog(@"Area = %f and perimeter = %f, circularity = %f, compactness = %f, shape index = %f effective diameter = %f",area,perimeter,circularity,compactness,shapeIndex,effectiveDiameter);
        
