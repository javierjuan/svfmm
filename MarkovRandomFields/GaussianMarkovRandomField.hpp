/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Gaussian Markov Random Field                                             *
***************************************************************************/

#ifndef GAUSSIANMARKOVRANDOMFIELD_HPP
#define GAUSSIANMARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "MarkovRandomField.hpp"
#include "../Constants.hpp"

using namespace Eigen;

/***************************** Gaussian Markov Random Field *****************************/

template<int Dimensions>
class GaussianMarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~GaussianMarkovRandomField() {}
    
    Model model() const { return GAUSSIAN; }
    
    GaussianMarkovRandomField<Dimensions>& operator = (const GaussianMarkovRandomField<Dimensions> &other) {}
    
protected:
    GaussianMarkovRandomField() {}
    GaussianMarkovRandomField(const GaussianMarkovRandomField<Dimensions> &other) {}
};

/***************************** Isotropic Gaussian Markov Random Field *****************************/

template<int Dimensions>
class IsotropicGaussianMarkovRandomField : public IsotropicMarkovRandomField<Dimensions>, public GaussianMarkovRandomField<Dimensions>
{
public:
    virtual ~IsotropicGaussianMarkovRandomField() {}
    
    ArrayXd sigma() const { return m_sigma; }
    
    IsotropicGaussianMarkovRandomField<Dimensions>& operator = (const IsotropicGaussianMarkovRandomField<Dimensions> &other);
    
protected:
    IsotropicGaussianMarkovRandomField();
    IsotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    IsotropicGaussianMarkovRandomField(const IsotropicGaussianMarkovRandomField<Dimensions> &other);
    
    ArrayXd m_sigma;
};

/***************************** Implementation *****************************/

template<int Dimensions>
IsotropicGaussianMarkovRandomField<Dimensions>::IsotropicGaussianMarkovRandomField()
: IsotropicMarkovRandomField<Dimensions>(), GaussianMarkovRandomField<Dimensions>(), m_sigma(ArrayXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicGaussianMarkovRandomField<Dimensions>::IsotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: IsotropicMarkovRandomField<Dimensions>(mask, classes, topology), GaussianMarkovRandomField<Dimensions>(), m_sigma(ArrayXd(classes))
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicGaussianMarkovRandomField<Dimensions>::IsotropicGaussianMarkovRandomField(const IsotropicGaussianMarkovRandomField<Dimensions> &other)
: IsotropicMarkovRandomField<Dimensions>(other), GaussianMarkovRandomField<Dimensions>(other), m_sigma(other.sigma())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
IsotropicGaussianMarkovRandomField<Dimensions>& IsotropicGaussianMarkovRandomField<Dimensions>::operator = (const IsotropicGaussianMarkovRandomField<Dimensions> &other)
{
    IsotropicMarkovRandomField<Dimensions>::operator=(other);
    GaussianMarkovRandomField<Dimensions>::operator=(other);
    
    m_sigma = other.sigma();
    
    return *this;
}

/***************************** Anisotropic Gaussian Markov Random Field *****************************/

template<int Dimensions>
class AnisotropicGaussianMarkovRandomField : public AnisotropicMarkovRandomField<Dimensions>, public GaussianMarkovRandomField<Dimensions>
{
public:
    virtual ~AnisotropicGaussianMarkovRandomField() {}
    
    ArrayXXd sigma() const { return m_sigma; }
    
    AnisotropicGaussianMarkovRandomField<Dimensions>& operator = (const AnisotropicGaussianMarkovRandomField<Dimensions> &other);
    
protected:
    AnisotropicGaussianMarkovRandomField();
    AnisotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    AnisotropicGaussianMarkovRandomField(const AnisotropicGaussianMarkovRandomField<Dimensions> &other);
    
    ArrayXXd m_sigma;
};

/***************************** Implementation *****************************/

template<int Dimensions>
AnisotropicGaussianMarkovRandomField<Dimensions>::AnisotropicGaussianMarkovRandomField()
: AnisotropicMarkovRandomField<Dimensions>(), GaussianMarkovRandomField<Dimensions>(), m_sigma(ArrayXXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicGaussianMarkovRandomField<Dimensions>::AnisotropicGaussianMarkovRandomField(const cimg_library::CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: AnisotropicMarkovRandomField<Dimensions>(mask, classes, topology), GaussianMarkovRandomField<Dimensions>(), m_sigma(ArrayXXd(classes, this->m_directions))
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicGaussianMarkovRandomField<Dimensions>::AnisotropicGaussianMarkovRandomField(const AnisotropicGaussianMarkovRandomField<Dimensions> &other)
: AnisotropicMarkovRandomField<Dimensions>(other), GaussianMarkovRandomField<Dimensions>(other), m_sigma(other.sigma())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
AnisotropicGaussianMarkovRandomField<Dimensions>& AnisotropicGaussianMarkovRandomField<Dimensions>::operator = (const AnisotropicGaussianMarkovRandomField<Dimensions> &other)
{
    AnisotropicMarkovRandomField<Dimensions>::operator=(other);
    GaussianMarkovRandomField<Dimensions>::operator=(other);
    
    m_sigma = other.sigma();
    
    return *this;
}

#endif