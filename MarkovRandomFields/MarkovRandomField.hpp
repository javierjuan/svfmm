/***************************************************************************
/* Javier Juan Albarracin - jajuaal1@ibime.upv.es                         */
/* Universidad Politecnica de Valencia, Spain                             */
/*                                                                        */
/* Copyright (C) 2020 Javier Juan Albarracin                              */
/*                                                                        */
/***************************************************************************
* Markov Random Field                                                      *
***************************************************************************/

#ifndef MARKOVRANDOMFIELD_HPP
#define MARKOVRANDOMFIELD_HPP

#define cimg_display 0
#include <CImg.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include "../Constants.hpp"

using namespace cimg_library;
using namespace Eigen;

/***************************** Markov Random Field Base *****************************/

class MarkovRandomFieldBase
{
public:
    enum Topology { ORTHOGONAL, COMPLETE };
    enum Tropism { ISOTROPIC, ANISOTROPIC };
    enum Model { GAUSSIAN, TSTUDENT, NONLOCAL };
    enum Estimation { FREQUENTIST, BAYESIAN };
    
    virtual Topology topology() const = 0;
    virtual Tropism tropism() const = 0;
    virtual Model model() const = 0;
    virtual Estimation estimation() const = 0;
};

/***************************** Markov Random Field *****************************/

template<int Dimensions>
class MarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~MarkovRandomField() {}
    
    int cliques() const { return m_cliques; }
    int connectivity() const { return m_connectivity; }
    int nodes() const { return m_nodes; }
    int classes() const { return m_classes; }
    std::vector<int> neighbours() const { return m_neighbours; }
    
    Topology topology() const { return m_topology; }
    
    bool isEmpty() const { return m_cliques == 0 || m_connectivity == 0 || m_nodes == 0 || m_classes == 0 || m_coefficients.size() == 0 || m_neighbours.size() == 0; }
    bool isCompatible(const MarkovRandomField<Dimensions> &markovRandomField) const;
    
    void addClassToMarkovRandomField(const ArrayXd &coefficients);
    
    ArrayXXd coefficients() const { return m_coefficients; }
    ArrayXd classCoefficients(const int index) const { return m_coefficients.col(index); }
    ArrayXd nodeCoefficients(const int index) const { return m_coefficients.row(index); }
    
    void setCoefficients(const ArrayXXd &coefficients);
    void setClassCoefficients(const ArrayXd &coefficients, const int index);
    void setNodeCoefficients(const ArrayXd &coefficients, const int index);
    
    Vector3i linearIndex2Coordinates(const CImg<bool> &image, const int index) const;
    int coordinates2LinearIndex(const CImg<bool> &mask, const int i, const int j, const int k = 0) const;

    int operator [] (const int i) const { return m_neighbours[i]; }
    MarkovRandomField<Dimensions>& operator = (const MarkovRandomField<Dimensions> &other);
    
    virtual void initialization(const ArrayXXd &coefficients) = 0;
    virtual void updateDistributions() = 0;
    virtual void updateParameters(const ArrayXXd &posteriorProbabilities) = 0;
    virtual double logLikelihood() = 0;

    virtual std::unique_ptr<MarkovRandomField<Dimensions>> clone() const = 0;

protected:
    MarkovRandomField();
    MarkovRandomField(const int cliques, const int connectivity, const int nodes, const int classes, const Topology topology);
    MarkovRandomField(const MarkovRandomField<Dimensions> &other);
    
    void build(const CImg<bool> &mask, const Topology topology);
    
    virtual void orthogonal(const CImg<bool> &mask) = 0;
    virtual void complete(const CImg<bool> &mask) = 0;
    CImg<int> mapLinearIndices2Coordinates(const CImg<bool> &mask);
    
    bool cuadraticSolver(const double a, const double b, const double c, double &r1, double &r2) const;
    void cubicSolver(const double a, const double b, const double c, const double d, double &r1, double &r2, double &r3, double &i1, double &i2, double &i3, double &h) const;
    ArrayXd conjugateProjection(const ArrayXd &a, const ArrayXd &weights) const;

    int m_cliques;
    int m_connectivity;
    int m_nodes;
    int m_classes;
    ArrayXXd m_coefficients;
    std::vector<int> m_neighbours;
    Topology m_topology;
};

/***************************** Implementation *****************************/

template<int Dimensions>
MarkovRandomField<Dimensions>::MarkovRandomField()
: m_cliques(0), m_connectivity(0), m_nodes(0), m_classes(0), m_coefficients(ArrayXXd()), m_neighbours(std::vector<int>()), m_topology(COMPLETE)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
MarkovRandomField<Dimensions>::MarkovRandomField(const int cliques, const int connectivity, const int nodes, const int classes, const Topology topology)
: m_cliques(cliques), m_connectivity(connectivity), m_nodes(nodes), m_classes(classes), m_coefficients(ArrayXXd(nodes, classes)), m_neighbours(std::vector<int>()), m_topology(topology)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
MarkovRandomField<Dimensions>::MarkovRandomField(const MarkovRandomField<Dimensions> &other)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_cliques = other.cliques();
    m_connectivity = other.connectivity();
    m_nodes = other.nodes();
    m_classes = other.classes();
    m_coefficients = other.coefficients();
    m_neighbours = other.neighbours();
    m_topology = other.topology();
}

template<int Dimensions>
MarkovRandomField<Dimensions>& MarkovRandomField<Dimensions>::operator = (const MarkovRandomField<Dimensions> &other)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_cliques = other.cliques();
    m_connectivity = other.connectivity();
    m_nodes = other.nodes();
    m_classes = other.classes();
    m_coefficients = other.coefficients();
    m_neighbours = other.neighbours();
    m_topology = other.topology();

    return *this;
}

template<int Dimensions>
bool MarkovRandomField<Dimensions>::isCompatible(const MarkovRandomField<Dimensions> &markovRandomField) const
{
    return m_nodes == markovRandomField.nodes() && m_classes == markovRandomField.classes() &&
           m_coefficients.rows() == markovRandomField.coefficients().rows() && m_coefficients.cols() == markovRandomField.coefficients().cols() &&
           m_neighbours.size() == markovRandomField.neighbours().size();
}

template<int Dimensions>
void MarkovRandomField<Dimensions>::addClassToMarkovRandomField(const ArrayXd &coefficients)
{
    if (coefficients.size() != m_nodes || coefficients.size() != m_coefficients.rows())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    ++m_classes;
    m_coefficients.conservativeResize(m_nodes, m_classes);
    m_coefficients.col(m_classes - 1) = coefficients;
}

template<int Dimensions>
void MarkovRandomField<Dimensions>::setCoefficients(const ArrayXXd &coefficients)
{
    if (coefficients.rows() != m_nodes || coefficients.cols() != m_classes)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Inconsistent coefficients" << std::endl;
        throw std::runtime_error(s.str());
    }

    m_coefficients = coefficients;
}

template<int Dimensions>
void MarkovRandomField<Dimensions>::setClassCoefficients(const ArrayXd &coefficients, const int index)
{
    if (index < 0 || index >= m_coefficients.cols())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Index out of range" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (coefficients.size() != m_nodes || coefficients.size() != m_coefficients.rows())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid coefficients" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_coefficients.col(index) = coefficients;
}

template<int Dimensions>
void MarkovRandomField<Dimensions>::setNodeCoefficients(const ArrayXd &coefficients, const int index)
{
    if (index < 0 || index >= m_coefficients.rows())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Index out of range" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (coefficients.size() != m_classes || coefficients.size() != m_coefficients.cols())
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Invalid coefficients" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    if (coefficients.sum() != 1)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Coefficients must sum 1" << std::endl;
        throw std::runtime_error(s.str());
    }
    
    m_coefficients.row(index) = coefficients;
}

template<int Dimensions>
void MarkovRandomField<Dimensions>::build(const CImg<bool> &mask, const Topology topology)
{
    switch (topology)
    {
    case ORTHOGONAL:
        orthogonal(mask);
        break;

    case COMPLETE:
    default:
        complete(mask);
        break;
    }
    m_coefficients = ArrayXXd::Ones(m_nodes, m_classes) / m_classes;
}

template<int Dimensions>
CImg<int> MarkovRandomField<Dimensions>::mapLinearIndices2Coordinates(const CImg<bool> &mask)
{
    m_nodes = 0;
    CImg<int> map(mask.width(), mask.height(), mask.depth());
    for (int x = 0; x < mask.width(); ++x)
    {
        for (int y = 0; y < mask.height(); ++y)
        {
            for (int z = 0; z < mask.depth(); ++z)
            {
                if (!mask(x, y, z))
                {
                    map(x, y, z) = -1;
                    continue;
                }

                map(x, y, z) = m_nodes;
                ++m_nodes;
            }
        }
    }
    return map;
}

template<int Dimensions>
Vector3i MarkovRandomField<Dimensions>::linearIndex2Coordinates(const CImg<bool> &mask, const int index) const
{
    int _index = 0;
    Vector3i coordinates;
    coordinates << -1, -1, -1;
    for (int x = 0; x < mask.width(); ++x)
    {
        for (int y = 0; y < mask.height(); ++y)
        {
            for (int z = 0; z < mask.depth(); ++z)
            {
                if (!mask(x, y, z))
                    continue;

                if (_index == index)
                {
                    coordinates(0) = x;
                    coordinates(1) = y;
                    coordinates(2) = z;
                    return coordinates;
                }
                ++_index;
            }
        }
    }
    return coordinates;
}

template<int Dimensions>
int MarkovRandomField<Dimensions>::coordinates2LinearIndex(const CImg<bool> &mask, const int i, const int j, const int k) const
{
    int index = -1;
    for (int x = 0; x < mask.width(); ++x)
    {
        for (int y = 0; y < mask.height(); ++y)
        {
            for (int z = 0; z < mask.depth(); ++z)
            {
                if (!mask(x, y, z))
                    continue;

                if (x == i && y == j && z == k)
                    return index;
                ++index;
            }
        }
    }
    return index;
}

template<int Dimensions>
bool MarkovRandomField<Dimensions>::cuadraticSolver(const double a, const double b, const double c, double &r1, double &r2) const
{
    const double s = (b * b) - (4.0 * a * c);
    if (s < 0)
        return false;

    const double sqrts = sqrt(s);
    const double a2 = 2.0 * a;
    r1 = (-b + sqrts) / a2;
    r2 = (-b - sqrts) / a2;
    return true;
}

template<int Dimensions>
void MarkovRandomField<Dimensions>::cubicSolver(const double a, const double b, const double c, const double d, double &r1, double &r2, double &r3, double &i1, double &i2, double &i3, double &h) const
{
    // First change the cubic equation to a depressed cubic formula: x^3 + fx + g = 0
    const double f = ((3.0 * a * c) - (b * b)) / (3.0 * a * a);
    const double g = ((2.0 * b * b * b) - (9.0 * a * b * c) + (27.0 * a * a * d)) / (27.0 * a * a * a);
    h = ((g * g) / 4.0) + ((f * f * f) / 27.0);

    // There is only one real root
    if (h > 0)
    {
        const double r = -(g / 2.0) + sqrt(h);
        const double s = r < 0 ? -pow(-r, 1.0 / 3.0) : pow(r, 1.0 / 3.0);
        const double t = -(g / 2.0) - sqrt(h);
        const double u = t < 0 ? -pow(-t, 1.0 / 3.0) : pow(t, 1.0 / 3.0);
        r1 = (s + u) - (b / (3.0 * a));
        r2 = -((s + u) / 2.0) - (b / (3.0 * a));
        i2 = ((s - u) * SVFMM_SQRT3) / 2.0;
        r3 = r2;
        i3 = -i2;
    }
    else
    {
        // All three roots are real and equal
        if (f == 0 && g == 0 && h == 0)
        {
            r1 = -pow((d / a), 1.0 / 3.0);
            r2 = r1;
            r3 = r1;
        }
        // All three roots are real
        else
        {
            const double i = sqrt(((g * g) / 4.0) - h);
            const double j = pow(i, 1.0 / 3.0);
            const double k = acos(-(g / (2.0 * i)));
            const double l = -j;
            const double m = cos(k / 3.0);
            const double n = SVFMM_SQRT3 * sin(k / 3.0);
            const double p = -(b / (3.0 * a));
            r1 = (2.0 * j * cos(k / 3.0)) - (b / (3.0 * a));
            r2 = (l * (m + n)) + p;
            r3 = (l * (m - n)) + p;
        }
    }
}

template<int Dimensions>
ArrayXd MarkovRandomField<Dimensions>::conjugateProjection(const ArrayXd &a, const ArrayXd &weights) const
{
    // Quadratic convex programming solution proposed in <A Spatially-Constrained Mixture Model for Image Segmentation>
    // Improved by the possibility of passing weights for each component of <a>
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;

    // Check if a has too large variates
    int aMaxIdx = 0;
    const double aMax = a.maxCoeff(&aMaxIdx);
    double aSumAbs = 0;
    for (int i = 0; i < m_classes; ++i)
        aSumAbs += std::fabs(a(i));
    if (aMax > 0 && (aSumAbs - aMax) < 1e-5)
    {
        ArrayXd y(m_classes);
        for (int i = 0; i < m_classes; ++i)
            y(i) = SVFMM_EPS;
        y(aMaxIdx) = 1.0 - (SVFMM_EPS * (m_classes - 1));
        return y;
    }

    // Conjugate projection
    double aSum = 0;
    double wSum = 0;
    for (int i = 0; i < m_classes; ++i)
    {
        aSum += a(i);
        wSum += (1.0 / weights(i));
    }

    ArrayXd y(m_classes);
    ArrayXd g(m_classes);
    ArrayXd yInit(m_classes);
    ArrayXd lambda(m_classes);
    ArrayXb activeSet(m_classes);

    for (int i = 0; i < m_classes; ++i)
    {
        g(i) = -1.0 / (wSum * weights(i));
        y(i) = a(i) - g(i) + (g(i) * aSum);
        yInit(i) = y(i);
        lambda(i) = 0;
        activeSet(i) = false;
    }

    while (true)
    {
        int m = 0;
        for (int i = 0; i < m_classes; ++i)
        {
            if (y(i) < SVFMM_EPS)
            {
                activeSet(i) = true;
                ++m;
            }
        }
        if (m == 0)
            break;

        double sumGActive = 0;
        double sumGInactive = 0;
        for (int i = 0; i < m_classes; ++i)
        {
            if (activeSet(i))
                sumGActive += g(i) * yInit(i);
            else
                sumGInactive += g(i);
            lambda(i) = 0;
        }

        double lSum = 0;
        for (int i = 0; i < m_classes; ++i)
        {
            if (activeSet(i))
                lambda(i) = -yInit(i) - (sumGActive / sumGInactive);
            else
                lambda(i) = 0;

            lSum += lambda(i);
        }
        for (int i = 0; i < m_classes; ++i)
        {
            if (activeSet(i))
                y(i) = SVFMM_EPS;
            else
                y(i) = a(i) - g(i) + (g(i) * aSum) + (g(i) * lSum) + lambda(i);
        }
    }
    return y;
}

/***************************** IsotropicMarkovRandomField *****************************/

template<int Dimensions>
class IsotropicMarkovRandomField : public MarkovRandomField<Dimensions>
{
public:
    virtual ~IsotropicMarkovRandomField() {}

    MarkovRandomFieldBase::Tropism tropism() const { return MarkovRandomFieldBase::ISOTROPIC; }
    
    int neighbour(const int i, const int n) const { return this->m_neighbours[i * this->m_connectivity + n]; }
    int operator () (const int i, const int n) const { return this->m_neighbours[i * this->m_connectivity + n]; }
    IsotropicMarkovRandomField<Dimensions>& operator = (const IsotropicMarkovRandomField<Dimensions> &other);
    
protected:
    IsotropicMarkovRandomField();
    IsotropicMarkovRandomField(const CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    IsotropicMarkovRandomField(const IsotropicMarkovRandomField<Dimensions> &other);

private:
    void orthogonal(const CImg<bool> &mask);
    void complete(const CImg<bool> &mask);
};

/***************************** Implementation *****************************/

template<int Dimensions>
IsotropicMarkovRandomField<Dimensions>::IsotropicMarkovRandomField()
: MarkovRandomField<Dimensions>()
{
}

template<int Dimensions>
IsotropicMarkovRandomField<Dimensions>::IsotropicMarkovRandomField(const CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: MarkovRandomField<Dimensions>(0, 0, 0, classes, topology)
{
    this->build(mask, topology);
}

template<int Dimensions>
IsotropicMarkovRandomField<Dimensions>::IsotropicMarkovRandomField(const IsotropicMarkovRandomField<Dimensions> &other)
: MarkovRandomField<Dimensions>(other)
{
}

template<int Dimensions>
IsotropicMarkovRandomField<Dimensions>& IsotropicMarkovRandomField<Dimensions>::operator = (const IsotropicMarkovRandomField<Dimensions> &other)
{
    MarkovRandomField<Dimensions>::operator=(other);
    
    return *this;
}

template<int Dimensions>
void IsotropicMarkovRandomField<Dimensions>::orthogonal(const CImg<bool> &mask)
{
    CImg<int> map(this->mapLinearIndices2Coordinates(mask));
    this->m_connectivity = Dimensions == 2 ? 4 : 6;
    this->m_cliques = this->m_connectivity;
    this->m_neighbours.resize(this->m_nodes * this->m_connectivity);

    int i = 0;
    const int width = mask.width();
    const int height = mask.height();
    const int depth = mask.depth();

    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                if (!mask(x, y, z))
                    continue;

                const int idx = i * this->m_connectivity;

                const int xm = x - 1 < 0       ? x : x - 1;
                const int xp = x + 1 >= width  ? x : x + 1;
                const int ym = y - 1 < 0       ? y : y - 1;
                const int yp = y + 1 >= height ? y : y + 1;

                this->m_neighbours[idx + 0] = mask(xm, y, z) ? map(xm, y, z) : map(x, y, z);
                this->m_neighbours[idx + 1] = mask(xp, y, z) ? map(xp, y, z) : map(x, y, z);

                this->m_neighbours[idx + 2] = mask(x, ym, z) ? map(x, ym, z) : map(x, y, z);
                this->m_neighbours[idx + 3] = mask(x, yp, z) ? map(x, yp, z) : map(x, y, z);

                if (Dimensions == 3)
                {
                    const int zm = z - 1 < 0      ? z : z - 1;
                    const int zp = z + 1 >= depth ? z : z + 1;

                    this->m_neighbours[idx + 4] = mask(x, y, zm) ? map(x, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + 5] = mask(x, y, zp) ? map(x, y, zp) : map(x, y, z);
                }
                ++i;
            }
        }
    }
}

template<int Dimensions>
void IsotropicMarkovRandomField<Dimensions>::complete(const CImg<bool> &mask)
{
    CImg<int> map(this->mapLinearIndices2Coordinates(mask));
    this->m_connectivity = Dimensions == 2 ? 8 : 26;
    this->m_cliques = this->m_connectivity;
    this->m_neighbours.resize(this->m_nodes * this->m_connectivity);

    int i = 0;
    const int width = mask.width();
    const int height = mask.height();
    const int depth = mask.depth();

    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                if (!mask(x, y, z))
                    continue;

                const int idx = i * this->m_connectivity;

                const int xm = x - 1 < 0       ? x : x - 1;
                const int xp = x + 1 >= width  ? x : x + 1;
                const int ym = y - 1 < 0       ? y : y - 1;
                const int yp = y + 1 >= height ? y : y + 1;

                this->m_neighbours[idx + 0] = mask(xm, y, z) ? map(xm, y, z) : map(x, y, z);
                this->m_neighbours[idx + 1] = mask(xp, y, z) ? map(xp, y, z) : map(x, y, z);

                this->m_neighbours[idx + 2] = mask(x, ym, z) ? map(x, ym, z) : map(x, y, z);
                this->m_neighbours[idx + 3] = mask(x, yp, z) ? map(x, yp, z) : map(x, y, z);

                this->m_neighbours[idx + 4] = mask(xm, ym, z) ? map(xm, ym, z) : map(x, y, z);
                this->m_neighbours[idx + 5] = mask(xp, yp, z) ? map(xp, yp, z) : map(x, y, z);

                this->m_neighbours[idx + 6] = mask(xp, ym, z) ? map(xp, ym, z) : map(x, y, z);
                this->m_neighbours[idx + 7] = mask(xm, yp, z) ? map(xm, yp, z) : map(x, y, z);

                if (Dimensions == 3)
                {
                    const int zm = z - 1 < 0      ? z : z - 1;
                    const int zp = z + 1 >= depth ? z : z + 1;

                    this->m_neighbours[idx + 8] = mask(x, y, zm) ? map(x, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + 9] = mask(x, y, zp) ? map(x, y, zp) : map(x, y, z);

                    this->m_neighbours[idx + 10] = mask(xm, y, zm) ? map(xm, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + 11] = mask(xp, y, zp) ? map(xp, y, zp) : map(x, y, z);

                    this->m_neighbours[idx + 12] = mask(xm, ym, zm) ? map(xm, ym, zm) : map(x, y, z);
                    this->m_neighbours[idx + 13] = mask(xp, yp, zp) ? map(xp, yp, zp) : map(x, y, z);

                    this->m_neighbours[idx + 14] = mask(x, ym, zm) ? map(x, ym, zm) : map(x, y, z);
                    this->m_neighbours[idx + 15] = mask(x, yp, zp) ? map(x, yp, zp) : map(x, y, z);

                    this->m_neighbours[idx + 16] = mask(xp, ym, zm) ? map(xp, ym, zm) : map(x, y, z);
                    this->m_neighbours[idx + 17] = mask(xm, yp, zp) ? map(xm, yp, zp) : map(x, y, z);

                    this->m_neighbours[idx + 18] = mask(xp, y, zm) ? map(xp, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + 19] = mask(xm, y, zp) ? map(xm, y, zp) : map(x, y, z);

                    this->m_neighbours[idx + 20] = mask(xp, yp, zm) ? map(xp, yp, zm) : map(x, y, z);
                    this->m_neighbours[idx + 21] = mask(xm, ym, zp) ? map(xm, ym, zp) : map(x, y, z);

                    this->m_neighbours[idx + 22] = mask(x, yp, zm) ? map(x, yp, zm) : map(x, y, z);
                    this->m_neighbours[idx + 23] = mask(x, ym, zp) ? map(x, ym, zp) : map(x, y, z);

                    this->m_neighbours[idx + 24] = mask(xm, yp, zm) ? map(xm, yp, zm) : map(x, y, z);
                    this->m_neighbours[idx + 25] = mask(xp, ym, zp) ? map(xp, ym, zp) : map(x, y, z);
                }
                ++i;
            }
        }
    }
}

/***************************** AnisotropicMarkovRandomField *****************************/

template<int Dimensions>
class AnisotropicMarkovRandomField : public MarkovRandomField<Dimensions>
{
public:
    virtual ~AnisotropicMarkovRandomField() {}
    
    MarkovRandomFieldBase::Tropism tropism() const { return MarkovRandomFieldBase::ANISOTROPIC; }
    int directions() const { return m_directions; }
    
    int neighbour(const int i, const int d, const int n) const { return this->m_neighbours[i * this->m_connectivity + d * this->m_cliques + n]; }
    int operator () (const int i, const int d, const int n) const { return this->m_neighbours[i * this->m_connectivity + d * this->m_cliques + n]; }
    AnisotropicMarkovRandomField<Dimensions> & operator = (const AnisotropicMarkovRandomField<Dimensions> &other);
    
protected:
    AnisotropicMarkovRandomField();
    AnisotropicMarkovRandomField(const CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology);
    AnisotropicMarkovRandomField(const AnisotropicMarkovRandomField<Dimensions> &other);
    
    int m_directions;

private:
    void orthogonal(const CImg<bool> &mask);
    void complete(const CImg<bool> &mask);
};

/***************************** Implementation *****************************/

template<int Dimensions>
AnisotropicMarkovRandomField<Dimensions>::AnisotropicMarkovRandomField()
: MarkovRandomField<Dimensions>(), m_directions(0)
{
}

template<int Dimensions>
AnisotropicMarkovRandomField<Dimensions>::AnisotropicMarkovRandomField(const CImg<bool> &mask, const int classes, const MarkovRandomFieldBase::Topology topology)
: MarkovRandomField<Dimensions>(0, 0, 0, classes, topology), m_directions(0)
{
    this->build(mask, topology);
}

template<int Dimensions>
AnisotropicMarkovRandomField<Dimensions>::AnisotropicMarkovRandomField(const AnisotropicMarkovRandomField<Dimensions> &other)
: MarkovRandomField<Dimensions>(other), m_directions(other.directions())
{
}

template<int Dimensions>
AnisotropicMarkovRandomField<Dimensions>& AnisotropicMarkovRandomField<Dimensions>::operator = (const AnisotropicMarkovRandomField<Dimensions> &other)
{
    MarkovRandomField<Dimensions>::operator=(other);
    m_directions = other.directions();

    return *this;
}

template<int Dimensions>
void AnisotropicMarkovRandomField<Dimensions>::orthogonal(const CImg<bool> &mask)
{
    CImg<int> map(this->mapLinearIndices2Coordinates(mask));
    this->m_connectivity = Dimensions == 2 ? 4 : 6;
    this->m_cliques = 2;
    this->m_directions = Dimensions == 2 ? 2 : 3;
    this->m_neighbours.resize(this->m_nodes * this->m_connectivity);

    int i = 0;
    const int width = mask.width();
    const int height = mask.height();
    const int depth = mask.depth();

    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                if (!mask(x, y, z))
                    continue;

                const int idx = i * this->m_connectivity;

                const int xm = x - 1 < 0       ? x : x - 1;
                const int xp = x + 1 >= width  ? x : x + 1;
                const int ym = y - 1 < 0       ? y : y - 1;
                const int yp = y + 1 >= height ? y : y + 1;
                
                this->m_neighbours[idx + (0 * this->m_cliques) + 0] = mask(xm, y, z) ? map(xm, y, z) : map(x, y, z);
                this->m_neighbours[idx + (0 * this->m_cliques) + 1] = mask(xp, y, z) ? map(xp, y, z) : map(x, y, z);

                this->m_neighbours[idx + (1 * this->m_cliques) + 0] = mask(x, ym, z) ? map(x, ym, z) : map(x, y, z);
                this->m_neighbours[idx + (1 * this->m_cliques) + 1] = mask(x, yp, z) ? map(x, yp, z) : map(x, y, z);

                if (Dimensions == 3)
                {
                    const int zm = z - 1 < 0      ? z : z - 1;
                    const int zp = z + 1 >= depth ? z : z + 1;

                    this->m_neighbours[idx + (2 * this->m_cliques) + 0] = mask(x, y, zm) ? map(x, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + (2 * this->m_cliques) + 1] = mask(x, y, zp) ? map(x, y, zp) : map(x, y, z);
                }
                ++i;
            }
        }
    }
}

template<int Dimensions>
void AnisotropicMarkovRandomField<Dimensions>::complete(const CImg<bool> &mask)
{
    CImg<int> map(this->mapLinearIndices2Coordinates(mask));
    this->m_connectivity = Dimensions == 2 ? 8 : 26;
    this->m_cliques = 2;
    this->m_directions = Dimensions == 2 ? 4 : 13;
    this->m_neighbours.resize(this->m_nodes * this->m_connectivity);

    int i = 0;
    const int width = mask.width();
    const int height = mask.height();
    const int depth = mask.depth();

    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int z = 0; z < depth; ++z)
            {
                if (!mask(x, y, z))
                    continue;

                const int idx = i * this->m_connectivity;

                const int xm = x - 1 < 0 ? x : x - 1;
                const int xp = x + 1 >= width ? x : x + 1;
                const int ym = y - 1 < 0 ? y : y - 1;
                const int yp = y + 1 >= height ? y : y + 1;

                this->m_neighbours[idx + (0 * this->m_cliques) + 0] = mask(xm, y, z) ? map(xm, y, z) : map(x, y, z);
                this->m_neighbours[idx + (0 * this->m_cliques) + 1] = mask(xp, y, z) ? map(xp, y, z) : map(x, y, z);

                this->m_neighbours[idx + (1 * this->m_cliques) + 0] = mask(x, ym, z) ? map(x, ym, z) : map(x, y, z);
                this->m_neighbours[idx + (1 * this->m_cliques) + 1] = mask(x, yp, z) ? map(x, yp, z) : map(x, y, z);

                this->m_neighbours[idx + (2 * this->m_cliques) + 0] = mask(xm, ym, z) ? map(xm, ym, z) : map(x, y, z);
                this->m_neighbours[idx + (2 * this->m_cliques) + 1] = mask(xp, yp, z) ? map(xp, yp, z) : map(x, y, z);

                this->m_neighbours[idx + (3 * this->m_cliques) + 0] = mask(xp, ym, z) ? map(xp, ym, z) : map(x, y, z);
                this->m_neighbours[idx + (3 * this->m_cliques) + 1] = mask(xm, yp, z) ? map(xm, yp, z) : map(x, y, z);

                if (Dimensions == 3)
                {
                    const int zm = z - 1 < 0      ? z : z - 1;
                    const int zp = z + 1 >= depth ? z : z + 1;

                    this->m_neighbours[idx + (4 * this->m_cliques) + 0] = mask(x, y, zm) ? map(x, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + (4 * this->m_cliques) + 1] = mask(x, y, zp) ? map(x, y, zp) : map(x, y, z);

                    this->m_neighbours[idx + (5 * this->m_cliques) + 0] = mask(xm, y, zm) ? map(xm, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + (5 * this->m_cliques) + 1] = mask(xp, y, zp) ? map(xp, y, zp) : map(x, y, z);

                    this->m_neighbours[idx + (6 * this->m_cliques) + 0] = mask(xm, ym, zm) ? map(xm, ym, zm) : map(x, y, z);
                    this->m_neighbours[idx + (6 * this->m_cliques) + 1] = mask(xp, yp, zp) ? map(xp, yp, zp) : map(x, y, z);

                    this->m_neighbours[idx + (7 * this->m_cliques) + 0] = mask(x, ym, zm) ? map(x, ym, zm) : map(x, y, z);
                    this->m_neighbours[idx + (7 * this->m_cliques) + 1] = mask(x, yp, zp) ? map(x, yp, zp) : map(x, y, z);

                    this->m_neighbours[idx + (8 * this->m_cliques) + 0] = mask(xp, ym, zm) ? map(xp, ym, zm) : map(x, y, z);
                    this->m_neighbours[idx + (8 * this->m_cliques) + 1] = mask(xm, yp, zp) ? map(xm, yp, zp) : map(x, y, z);

                    this->m_neighbours[idx + (9 * this->m_cliques) + 0] = mask(xp, y, zm) ? map(xp, y, zm) : map(x, y, z);
                    this->m_neighbours[idx + (9 * this->m_cliques) + 1] = mask(xm, y, zp) ? map(xm, y, zp) : map(x, y, z);

                    this->m_neighbours[idx + (10 * this->m_cliques) + 0] = mask(xp, yp, zm) ? map(xp, yp, zm) : map(x, y, z);
                    this->m_neighbours[idx + (10 * this->m_cliques) + 1] = mask(xm, ym, zp) ? map(xm, ym, zp) : map(x, y, z);

                    this->m_neighbours[idx + (11 * this->m_cliques) + 0] = mask(x, yp, zm) ? map(x, yp, zm) : map(x, y, z);
                    this->m_neighbours[idx + (11 * this->m_cliques) + 1] = mask(x, ym, zp) ? map(x, ym, zp) : map(x, y, z);

                    this->m_neighbours[idx + (12 * this->m_cliques) + 0] = mask(xm, yp, zm) ? map(xm, yp, zm) : map(x, y, z);
                    this->m_neighbours[idx + (12 * this->m_cliques) + 1] = mask(xp, ym, zp) ? map(xp, ym, zp) : map(x, y, z);
                }
                ++i;
            }
        }
    }
}

/***************************** Frequentist Markov Random Field *****************************/

template<int Dimensions>
class FrequentistMarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~FrequentistMarkovRandomField() {}

    Estimation estimation() const { return FREQUENTIST; }
    
    FrequentistMarkovRandomField<Dimensions>& operator = (const FrequentistMarkovRandomField<Dimensions> &other) {}

protected:
    FrequentistMarkovRandomField() {}
    FrequentistMarkovRandomField(const FrequentistMarkovRandomField<Dimensions> &other) {}
};

/***************************** Bayesian Markov Random Field *****************************/

template<int Dimensions>
class BayesianMarkovRandomField : public virtual MarkovRandomFieldBase
{
public:
    virtual ~BayesianMarkovRandomField() {}
    
    Estimation estimation() const { return BAYESIAN; } 
    ArrayXXd concentrations() const { return m_concentrations; }
    
    BayesianMarkovRandomField<Dimensions>& operator = (const BayesianMarkovRandomField<Dimensions> &other);
    
protected:
    BayesianMarkovRandomField();
    BayesianMarkovRandomField(const ArrayXXd &concentrations);
    BayesianMarkovRandomField(const BayesianMarkovRandomField<Dimensions> &other);
    
    ArrayXXd m_concentrations;
};

/***************************** Implementation *****************************/

template<int Dimensions>
BayesianMarkovRandomField<Dimensions>::BayesianMarkovRandomField()
: m_concentrations(ArrayXXd())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
BayesianMarkovRandomField<Dimensions>::BayesianMarkovRandomField(const ArrayXXd &concentrations)
: m_concentrations(concentrations)
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
BayesianMarkovRandomField<Dimensions>::BayesianMarkovRandomField(const BayesianMarkovRandomField<Dimensions> &other)
: m_concentrations(other.concentrations())
{
    if (Dimensions < 2 || Dimensions > 3)
    {
        std::stringstream s;
        s << "In function " << __PRETTY_FUNCTION__ << " ==> Dimensions not supported" << std::endl;
        throw std::runtime_error(s.str());
    }
}

template<int Dimensions>
BayesianMarkovRandomField<Dimensions>& BayesianMarkovRandomField<Dimensions>::operator = (const BayesianMarkovRandomField<Dimensions> &other)
{
    m_concentrations = other.concentrations();
    
    return *this;
}

#endif