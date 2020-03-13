/***************************************************************************/
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                          */
/* Universidad Politecnica de Valencia, Spain                              */
/*                                                                         */
/* Copyright (C) 2020 Javier Juan Albarracin                               */
/*                                                                         */
/***************************************************************************/
/* Spatially Variant Finite Mixture Models                                 */
/***************************************************************************/

#define cimg_display 0
#include <CImg.h>
#include <MarkovRandomFields/FrequentistGaussMarkovRandomField.hpp>
#include <MarkovRandomFields/BayesianGaussMarkovRandomField.hpp>
#include <Distributions/MultivariateNormal.hpp>
#include <Distributions/MultivariateTStudent.hpp>
#include <Distributions/Gamma.hpp>
#include <SpatiallyVariantFiniteMixtureModel.hpp>
#include <ExpectationMaximization.hpp>
#include <PyCImg.hpp>
#include <PySVFMM.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <string>

namespace py = pybind11;
using namespace cimg_library;

template<typename T>
using ndarray = py::array_t<T, py::array::c_style | py::array::forcecast>;


std::string doc()
{
    std::stringstream ss;

    ss << "Spatially Varying Finite Mixture Models" << std::endl;
    ss << "---------------------------------------" << std::endl;

    ss << "USAGE:" << std::endl;
    ss << "\tgsvfmm(input, k, <option>=<value>)" << std::endl;

    ss << "COMMAND:" << std::endl;
    ss << "\tgsvfmm:" << std::endl;
    ss << "\t\tA spatially varying finite mixture model segmentation algorithm that introduces a continuous gaussian Markov Random Field (MRF) over the prior probabilities of the model to impose spatial constraints during the learning process." << std::endl;
    ss << "OPTIONS:" << std::endl;

    ss << "\tmask=<numpy.array>" << std::endl;
    ss << "\t\tThis option allows to specify a numpy array of the shame shape of the input to define the region of interest to be segmented." << std::endl;

    ss << "\ttopology='orthogonal'/'complete'" << std::endl;
    ss << "\t\tThis option fixes the topology of the MRF. For 2-dimensional images: ORTHOGONAL implements 4 clique connectivity and COMPLETE implements 8 clique connectivity. For 3-dimensional images: ORTHOGONAL implements 6 clique connectivity and COMPLETE implements 26 clique connectivity" << std::endl;

    ss << "\ttropism='isotropic'/'anisotropic'" << std::endl;
    ss << "\t\tThis option fixes the spatial uniformity of the Markov Random Field. An isotropic MRF assumes that all cliques have the same variance per class. An anisotropic MRF specifies directions within the neighbourhood to capture different variances per class." << std::endl;

    ss << "\tdistribution='gaussian'/'student'" << std::endl;
    ss << "\t\tThis option sets the distribution employed to model the data." << std::endl;

    ss << "\testimation='frequentist'/'estimation'" << std::endl;
    ss << "\t\tThis option sets the estimation method employed to estimate the contextual mixing coefficients whithin the MRF. If 'frequentist' option is employed, contextual mixing proportions are treated as parameters (convex quadratic programming is employed to ensure coefficients to sum 1). If 'estimation' option is employed, contextual mixing proportions are treated as random variables drawn from a Dirichlet Compound Multinomial distribution that always guarantees that coefficients sum 1." << std::endl;
    
    ss << "\tregularization=1e-7" << std::endl;
    ss << "\t\tThis option enables the Tikhonov regularization to avoid ill-conditioned covariance matrices. An <epsilon> value is added to the diagonal of all covariance matrices." << std::endl;

    ss << "\tInitialization='kmeans'/<numpy.array>" << std::endl;
    ss << "\t\tInitialization of the svfmm. User can provide an initial labelling image of the same dimensions than the input image. Otherwise, user can employ k-means to generate an initialization." << std::endl;

    ss << "\tmax_iterations=25" << std::endl;
    ss << "\t\tMaximum number of iterations for the Expectation Maximization algorithm." << std::endl;

    ss << "\tthreshold=0.025" << std::endl;
    ss << "\t\tThreshold employed for convergence. Ratio of change between iterations of Expectation Maximization." << std::endl;

    ss << "\tverbose=True/False" << std::endl;
    ss << "\t\tVerbose output." << std::endl;
    
    ss << "\thelp" << std::endl;
    ss << "\t\tShows this help." << std::endl;

    return ss.str();
}


template<typename Distribution, int Dimensions>
py::dict classify(const CImg<double> &image, const CImg<bool> &mask, const int k, const MarkovRandomFieldBase::Topology topology, const std::string &tropism, 
                  const std::string &estimation, const CImg<int> &initialLabels, const int maxIterations, const double threshold, const double regularization, const bool verbose)
{
    std::unique_ptr<MarkovRandomField<Dimensions>> spatialCoefficients(nullptr);
    std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>> svfmm(nullptr);
    try
    {
        if (tropism == "isotropic")
        {
            if (estimation == "frequentist")
                spatialCoefficients = std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistIsotropicGaussMarkovRandomField<Dimensions>(mask, k, topology));
            else
                spatialCoefficients = std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianIsotropicGaussMarkovRandomField<Dimensions>(mask, k, topology));
        }
        else
        {
            if (estimation == "frequentist")
                spatialCoefficients = std::unique_ptr<MarkovRandomField<Dimensions>>(new FrequentistAnisotropicGaussMarkovRandomField<Dimensions>(mask, k, topology));
            else
                spatialCoefficients = std::unique_ptr<MarkovRandomField<Dimensions>>(new BayesianAnisotropicGaussMarkovRandomField<Dimensions>(mask, k, topology));
        }
        
        svfmm = std::shared_ptr<SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>>(new SpatiallyVariantFiniteMixtureModel<Distribution, Dimensions>(std::move(spatialCoefficients)));
        
        ExpectationMaximization<Distribution, Dimensions> EM(maxIterations, threshold, regularization, verbose);
        EM.learn(svfmm, image, mask, k, initialLabels);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what();
        return py::dict();
    }

    const py::dict results = PySVFMM<Distribution, Dimensions>::toPython(svfmm.get(), mask, PySVFMMBase::COMPLETE);

    svfmm.reset();

    return results;
}


py::dict fit(const ndarray<double> &input, int k, const ndarray<bool> &mask, const ndarray<int> &initialization, const std::string &distribution, const std::string &topology,
             const std::string &tropism, const std::string &estimation, const int max_iterations, const double regularization, const double threshold, const bool verbose)
{
    const CImg<double> image = PyCImg::toCImg(input);
    
    CImg<bool> mask_ = PyCImg::toCImg(mask);
    std::string maskInitialization = "user provided mask";
    if (mask_.is_empty())
    {
        maskInitialization = "none";
        mask_ = CImg<bool>(image.width(), image.height(), image.depth(), 1);
    }

    if (!image.is_sameXYZ(mask_))
    {
        std::stringstream ss;
        ss << "In function " << __PRETTY_FUNCTION__ << " ==> Image and mask dimensions must agree." << std::endl;
        throw std::runtime_error(ss.str());
    }

    CImg<int> labels = PyCImg::toCImg(initialization);
    std::string initializationType = "user provided labels";
    if (labels.is_empty())
    {
        initializationType = "kmeans";
    }

    const int dimensions = image.depth() == 1 ? 2 : 3;
    const MarkovRandomFieldBase::Topology topology_ = topology == "orthogonal" ? MarkovRandomFieldBase::ORTHOGONAL : MarkovRandomFieldBase::COMPLETE;

    // Show SVFMM configuration
    if (verbose)
    {
        std::cout << std::endl;
        std::cout << "CONFIGURATION:" << std::endl;
        std::cout << "\tImage size: " << image.width() << "x" << image.height() << "x" << image.depth() << "x" << image.spectrum() << std::endl;
        std::cout << "\tMask: " << maskInitialization << std::endl;
        std::cout << "\tNumber of classes: " << k << std::endl;
        std::cout << "\tDimensionality: " << dimensions << std::endl;
        std::cout << "\tDistribution: "<< distribution << std::endl;
        std::cout << "\tMRF Topology: " << topology << std::endl;
        std::cout << "\tMRF Tropism: " << tropism << std::endl;
        std::cout << "\tMRF Estimation: " << estimation << std::endl;
        std::cout << "\tInitialization: " << initializationType << std::endl;
        std::cout << "\tMaximum iterations: " << max_iterations << std::endl;
        std::cout << "\tConvergence threshold: " << threshold << std::endl;
        std::cout << "\tTikhonov regularization: " << regularization << std::endl;
        std::cout << "\tVerbose: " << (verbose ? "True" : "False") << std::endl;
        std::cout << std::endl;
    }  

    // Call templated function
    if (dimensions == 3)
    {
        if (distribution == "gaussian")
            return classify<MultivariateNormal, 3>(image, mask_, k, topology_, tropism, estimation, labels, max_iterations, threshold, regularization, verbose);
        else if (distribution == "student")
            return classify<MultivariateTStudent, 3>(image, mask_, k, topology_, tropism, estimation, labels, max_iterations, threshold, regularization, verbose);
        else
            return classify<Gamma, 3>(image, mask_, k, topology_, tropism, estimation, labels, max_iterations, threshold, regularization, verbose);
    }
    else
    {
    
        if (distribution == "gaussian")
            return classify<MultivariateNormal, 2>(image, mask_, k, topology_, tropism, estimation, labels, max_iterations, threshold, regularization, verbose);
        else if (distribution == "student")
            return classify<MultivariateTStudent, 2>(image, mask_, k, topology_, tropism, estimation, labels, max_iterations, threshold, regularization, verbose);
        else
            return classify<Gamma, 2>(image, mask_, k, topology_, tropism, estimation, labels, max_iterations, threshold, regularization, verbose);
    }
    
    return py::dict();
}


PYBIND11_MODULE(gsvfmm, m)
{
    const ndarray<bool> mask;
    const ndarray<int> initialization;
    const std::string distribution = "gaussian";
    const std::string topology = "complete";
    const std::string tropism = "anisotropic";
    const std::string estimation = "bayesian";
    const int max_iterations = 50;
    const double regularization = 1e-10;
    const double threshold = 1e-5;
    const bool verbose = true;

    m.doc() = doc();

    m.def("fit", &fit,
        py::return_value_policy::move,
        py::arg("input"), py::arg("k"), py::arg("mask") = mask,
        py::arg("initialization") = initialization, py::arg("distribution") = distribution, 
        py::arg("topology") = topology, py::arg("tropism") = tropism,
        py::arg("estimation") = estimation, py::arg("max_iterations") = max_iterations, 
        py::arg("regularization") = regularization, py::arg("threshold") = threshold, py::arg("verbose") = verbose
    );
}
