#ifndef DATAM_VLESS_GLOBAL_H
#define DATAM_VLESS_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(DATAM_VLESS_LIBRARY)
#  define DATAM_VLESSSHARED_EXPORT Q_DECL_EXPORT
#else
#  define DATAM_VLESSSHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // DATAM_VLESS_GLOBAL_H
