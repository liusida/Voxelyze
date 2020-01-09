#if !defined(VX3_QUAT3D_H)
#define VX3_QUAT3D_H

template <typename T = double>
class Quat3D {
public:
	T w; //!< The current W value.
	T x; //!< The current X value.
	T y; //!< The current Y value.
	T z; //!< The current Z value.
};

#endif // VX3_QUAT3D_H
