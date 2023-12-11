#pragma once
#ifndef VECTORISATION_HPP
#define VECTORISATION_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


/// @brief This function draws the axis of the vectorisation based 
/// from segmentation
/// @param img Matrix to be drawn
/// @param p Center of the vector to drawn
/// @param q Tip of the vector to drawn
/// @param colour Color of the vector to drawn
/// @param scale Scale of the vector to drawn
inline void drawAxis(
    cv::Mat& img, 
    cv::Point p, 
    cv::Point q, 
    const cv::Scalar& colour, 
    const float scale = 0.2
)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, cv::LINE_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, cv::LINE_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, cv::LINE_AA);
}

/// @brief Get center of the segmentation mask
/// @param analysis PCA analysis to be used
/// @return A 2D point from OpenCV
inline cv::Point getCenter(const cv::PCA& analysis)
{
    return cv::Point(
        static_cast<int>(analysis.mean.at<double>(0, 0)),
        static_cast<int>(analysis.mean.at<double>(0, 1))
    );
}

/// @brief Get center of the segmentation mask
/// @param analysis PCA analysis to be used
/// @param center Center point where to store
inline void getCenter(const cv::PCA& analysis, cv::Point& center)
{
    center = cv::Point(
        static_cast<int>(analysis.mean.at<double>(0, 0)),
        static_cast<int>(analysis.mean.at<double>(0, 1))
    );
}

/// @brief Get the eigen value and eigen vectors from the PCA analysis.
/// The eigen vectors are the orientation vectors and the eigen values 
/// are the dilatation of the data
/// @param analysis PCA analysis to be used
/// @return (two eigen vector, two eigen values)
inline std::tuple<std::vector<cv::Point2d>, std::vector<double>> getEigens(
    const cv::PCA& analysis
)
{
    std::vector<cv::Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = cv::Point2d(analysis.eigenvectors.at<double>(i, 0),
                                analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = analysis.eigenvalues.at<double>(i);
    }
    return std::tuple(eigen_vecs, eigen_val);
}

/// @brief Get the eigen value and eigen vectors from the PCA analysis.
/// The eigen vectors are the orientation vectors and the eigen values 
/// are the dilatation of the data
/// @param analysis PCA analysis to be used
/// @param storing_tuple  (two eigen vector, two eigen values)
inline void getEigens(
    const cv::PCA& analysis,
    std::tuple<std::vector<cv::Point2d>, std::vector<double>> storing_tuple
)
{
    std::vector<cv::Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = cv::Point2d(analysis.eigenvectors.at<double>(i, 0),
                                analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = analysis.eigenvalues.at<double>(i);
    }
    storing_tuple = std::tuple(eigen_vecs, eigen_val);
}

/// @brief Performs a PCA analysis
/// @param pts Points to be analysed
inline cv::PCA getPCA(
    const std::vector<cv::Point>& pts
)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    return cv::PCA(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
}

/// @brief Performs a PCA analysis
/// @param pts Points to be analysed
/// @param analysis PCA class to store the analysis
inline void getPCA(
    const std::vector<cv::Point>& pts,
    cv::PCA& analysis
)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    analysis = cv::PCA(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
}

// inline std::vector<cv::Point> getPoints(const cv::PCA& analysis)
// {
// }

/// @brief Eigen vectors are the directions and eigen values are the repartition
/// @param pts Points from which to get the orientation
/// @return parrelle Point, ortho point, angle
std::tuple<cv::Point, cv::Point, double> getOrientation(
    const std::vector<cv::Point>& pts
);

/// @brief Eigen vectors are the directions and eigen values are the repartition
/// @param pts Points from which to get the orientation
/// @param storing_tuple parrelle Point, ortho point, angle
inline void getOrientation(
    const std::vector<cv::Point>& pts,
    std::tuple<cv::Point, cv::Point, double> storing_tuple
)
{
    storing_tuple = getOrientation(pts);
}

/// @brief Eigen vectors are the directions and eigen values are the repartition
/// @param pts points from which to get the orientation
/// @param Mat matrix on which printing the vectors
/// @return parrelle point, ortho point, angle
std::tuple<cv::Point, cv::Point, double> getOrientation(
    const std::vector<cv::Point> &pts, cv::Mat &img
);

/// @brief Eigen vectors are the directions and eigen values are the repartition
/// @param pts points from which to get the orientation
/// @param Mat matrix on which printing the vectors
/// @param storing_tuple point, ortho point, angle
inline void getOrientation(
    const std::vector<cv::Point> &pts, 
    cv::Mat &img,
    std::tuple<cv::Point, cv::Point, double> storing_tuple
)
{
    storing_tuple = getOrientation(pts, img);
}

#endif // VECTORISATION_HPP