# include <stdio.h>
# include <stdlib.h>
# include <string.h>

#define MIN_LEN   256
#define CVEFAILED  -1
#define CVESUCCESS  0
#define CVEPUSHBACK 1
#define CVEPOPBACK  2
#define CVEINSERT   3
#define CVERM		4
#define EXPANED_VAL 1
#define REDUSED_VAL 2

typedef void *citerator;
typedef struct _cvector
{
	void *cv_pdata;
	size_t cv_len, cv_tot_len, cv_size;
} *cvector;

# define CWARNING_ITER(cv, iter, file, func, line) \
do {\
if ((cvector_begin(cv) > iter) || (cvector_end(cv) <= iter)) { \
	fprintf(stderr, "var(" iter ") warng out of range, "\
		"at file:%s func:%s line:%d!!/n", file, func, line); \
		return CVEFAILED; \
} \
} while (0)

# if def _cplusplus
# define EXTERN_ extern "C"
# else
# define EXTERN_ extern
# endif

//EXTERN_ cvector		cvector_create(const size_t size);
//EXTERN_ void		cvector_destroy(const cvector cv);
//EXTERN_ size_t		cvector_length(const cvector cv);
//EXTERN_ int			cvector_pushback(const cvector cv, void *memb);
//EXTERN_	int			cvector_popback(const cvector cv, void *memb);
//EXTERN_	size_t		cvector_iter_at(const cvector cv, citerator iter);
//EXTERN_ int			cvector_iter_val(const cvector cv, citerator iter, void *memb);
//EXTERN_ citerator	cvector_begin(const cvector cv);
//EXTERN_ citerator	cvector_end(const cvector cv);
//EXTERN_	citerator	cvector_next(const cvector cv, citerator iter);
//EXTERN_ int			cvector_val_at(const cvector cv, size_t index, void *memb);
//EXTERN_ int			cvector_insert(const cvector cv, citerator iter, void *memb);
//EXTERN_ int			cvector_insert_at(const cvector cv, size_t index, void *memb);
//EXTERN_	int			cvector_rm(const cvector cv, citerator iter);
//EXTERN_	int			cvector_rm_at(const cvector cv, size_t index);
//
///* for test */
//EXTERN_ void		cv_info(const cvector cv);
//EXTERN_ void		cv_print(const cvector cv);

cvector cvector_create(const size_t size)
{
	cvector cv = (cvector)malloc(sizeof(struct _cvector));

	if (!cv) return NULL;

	cv->cv_pdata = malloc(MIN_LEN * size);

	if (!cv->cv_pdata)
	{
		free(cv);
		return NULL;
	}

	cv->cv_size = size;
	cv->cv_tot_len = MIN_LEN;
	cv->cv_len = 0;

	return cv;
}

void cvector_destroy(const cvector cv)
{
	free(cv->cv_pdata);
	free(cv);
	return;
}

size_t cvector_length(const cvector cv)
{
	return cv->cv_len;
}

int cvector_pushback(const cvector cv, void *memb)
{
	if (cv->cv_len >= cv->cv_tot_len)
	{
		void *pd_sav = cv->cv_pdata;
		cv->cv_tot_len <<= EXPANED_VAL;
		cv->cv_pdata = realloc(cv->cv_pdata, cv->cv_tot_len * cv->cv_size);

		if (!cv->cv_pdata)
		{
			cv->cv_pdata = pd_sav;
			cv->cv_tot_len >>= EXPANED_VAL;
			return CVEPUSHBACK;
		}
	}

	memcpy((char *)cv->cv_pdata + cv->cv_len * cv->cv_size, memb, cv->cv_size);
	cv->cv_len++;

	return CVESUCCESS;
}

int cvector_popback(const cvector cv, void *memb)
{
	if (cv->cv_len <= 0)
	{
		return CVEPOPBACK;
	}

	cv->cv_len--;
	memcpy(memb, (char*)cv->cv_pdata + cv->cv_len * cv->cv_size, cv->cv_size);

	if ((cv->cv_tot_len >= (MIN_LEN << REDUSED_VAL))
		&& (cv->cv_len <= (cv->cv_tot_len >> REDUSED_VAL)))
	{
		void *pd_sav = cv->cv_pdata;
		cv->cv_tot_len >>= EXPANED_VAL;
		cv->cv_pdata = realloc(cv->cv_pdata, cv->cv_tot_len * cv->cv_size);

		if (!cv->cv_pdata)
		{
			cv->cv_tot_len <<= EXPANED_VAL;
			cv->cv_pdata = pd_sav;
			return CVEPOPBACK;
		}
	}

	return CVESUCCESS;
}

size_t cvector_iter_at(const cvector cv, citerator iter)
{
	return ((char *)iter - (char *)(cv->cv_pdata)) / cv->cv_size;
}

int cvector_iter_val(const cvector cv, citerator iter, void *memb)
{
	memcpy(memb, iter, cv->cv_size);
	return 0;
}

citerator cvector_begin(const cvector cv)
{
	return cv->cv_pdata;
}

citerator cvector_end(const cvector cv)
{
	return (char *)cv->cv_pdata + (cv->cv_size * cv->cv_len);
}

static inline void cvmemove_foreward(const cvector cv, void *from, void *to)
{
	size_t size = cv->cv_size;
	char *p;
	for (p = (char *)to; p >= from; p -= size)
	{
		memcpy(p + size, p, size);
	}

	return;
}

static inline void cvmemove_backward(const cvector cv, void *from, void *to)
{
	memcpy(from, (char *)from + cv->cv_size, (char *)to - (char *)from);
	return;
}

int cvector_insert(const cvector cv, citerator iter, void *memb)
{
	if (cv->cv_len >= cv->cv_tot_len)
	{
		void *pd_sav = cv->cv_pdata;
		cv->cv_tot_len <<= EXPANED_VAL;
		cv->cv_pdata = realloc(cv->cv_pdata, cv->cv_tot_len * cv->cv_size);

		if (!cv->cv_size)
		{
			cv->cv_pdata = pd_sav;
			cv->cv_tot_len >>= EXPANED_VAL;
			return CVEINSERT;
		}
	}

	cvmemove_foreward(cv, iter, (char *)cv->cv_pdata + cv->cv_len * cv->cv_size);
	memcpy(iter, memb, cv->cv_size);
	cv->cv_len++;

	return CVESUCCESS;
}

int cvector_insert_at(const cvector cv, size_t index, void *memb)
{
	citerator iter;

	if (index >= cv->cv_tot_len)
	{
		cv->cv_len = index + 1;
		while (cv->cv_len >= cv->cv_tot_len)
		{
			cv->cv_tot_len <<= EXPANED_VAL;
		}
		cv->cv_pdata = realloc(cv->cv_pdata, cv->cv_tot_len * cv->cv_size);
		iter = (char *)cv->cv_pdata + cv->cv_size * index;
		memcpy(iter, memb, cv->cv_size);
	}
	else
	{
		iter = (char *)cv->cv_pdata + cv->cv_size *index;
		cvector_insert(cv, iter, memb);
	}

	return 0;
}

citerator cvector_next(const cvector cv, citerator iter)
{
	return (char *)iter + cv->cv_size;
}

int cvector_val(const cvector cv, citerator iter, void *memb)
{
	memcpy(memb, iter, cv->cv_size);
	return 0;
}

int cvector_val_at(const cvector cv, size_t index, void *memb)
{
	memcpy(memb, (char *)cv->cv_pdata + index * cv->cv_size, cv->cv_size);
	return 0;
}

int cvector_rm(const cvector cv, citerator iter)
{
	citerator from;
	citerator end;

	from = iter;
	end = cvector_end(cv);
	memcpy(from, (char *)from + cv->cv_size, (char *)end - (char *)from);
	cv->cv_len--;

	if ((cv->cv_tot_len >= (MIN_LEN << REDUSED_VAL))
		&& (cv->cv_len <= (cv->cv_tot_len >> REDUSED_VAL)))
	{
		void *pd_sav = cv->cv_pdata;
		cv->cv_tot_len >>= EXPANED_VAL;
		cv->cv_pdata = realloc(cv->cv_pdata, cv->cv_tot_len * cv->cv_size);

		if (!cv->cv_pdata)
		{
			cv->cv_tot_len <<= EXPANED_VAL;
			cv->cv_pdata = pd_sav;
			return CVERM;
		}
	}

	return CVESUCCESS;
}

int cvector_rm_at(const cvector cv, size_t index)
{
	citerator iter;

	iter = (char *)cv->cv_pdata + cv->cv_size * index;
	return cvector_rm(cv, iter);
}

void cv_info(const cvector cv)
{
	return;
}

void cv_print(const cvector cv)
{
	int num;
	citerator iter;

	if (0 == cvector_length(cv))
	{
		printf("cvector_length is zero");
	}

	for (iter = cvector_begin(cv); iter != cvector_end(cv); iter = cvector_next(cv, iter))
	{
		cvector_iter_val(cv, iter, &num);
		printf("var:%d at:%d\n", num, cvector_iter_at(cv, iter));
	}

	return;
}