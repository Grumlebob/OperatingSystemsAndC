/*
A threadsafe linked-list implementation of a cache
 */

// Todo - can this be removed by importing from proxy.h?
#define MAX_CACHE_SIZE 1049000
#define MAX_OBJECT_SIZE 102400

typedef struct cache_block
{
    char *request_header;
    char *content;
    size_t size;
    struct cache_block *prev;
    struct cache_block *next;
} cache_block;

void init_cache();
void insert_head(char *request_header, char *content, size_t size);
void move_to_head(cache_block *block);
cache_block *find(char *request_header);