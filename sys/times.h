#ifndef _TIMES_H
#define _TIMES_H

#ifdef _WIN32
#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>

int gettimeofday(struct timeval* t, void* timezone)
{
	struct _timeb timebuffer;
	_ftime(&timebuffer);
	t->tv_sec = timebuffer.time;
	t->tv_usec = 1000 * timebuffer.millitm;
	return 0;
}

// from linux's sys/times.h

//#include <features.h>

#define __need_clock_t
#include <time.h>


/* Structure describing CPU time used by a process and its children.  */
struct tms
  {
    clock_t tms_utime;          /* User CPU time.  */
    clock_t tms_stime;          /* System CPU time.  */

    clock_t tms_cutime;         /* User CPU time of dead children.  */
    clock_t tms_cstime;         /* System CPU time of dead children.  */
  };

/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t times(struct tms *__buffer) {

	__buffer->tms_utime = clock();
	__buffer->tms_stime = 0;
	__buffer->tms_cstime = 0;
	__buffer->tms_cutime = 0;
	return __buffer->tms_utime;
}

typedef long long suseconds_t ;
#else
#include<sys/time.h>
#endif
#endif