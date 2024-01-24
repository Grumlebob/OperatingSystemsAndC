#include "malloc.h"
#include "errno.h"
#include "stdlib.h"
#include "cache.h"
#include "string.h"
#include "proxy.h"

// head.Previous is the tail of the list.
// Previous is also used, when removing the tail (because of LRU), then we need to make head.prev the new tail.
cache_block *head;

void init_cache()
{
    cache_block *start_cache = malloc(sizeof(cache_block));
    if (start_cache == NULL)
    {
        fprintf(stderr, "Error: Failed to initialize the cache.\n");
        exit(EXIT_FAILURE);
    }

    // Head is a header node with no payload.
    start_cache->request_header = NULL;
    start_cache->content = NULL;
    start_cache->prev = start_cache;
    start_cache->next = start_cache;
    start_cache->size = 0;

    head = start_cache;
}

void insert_head(char *header, char *content, size_t size)
{
    cache_block *new_block;

    if ((new_block = malloc(sizeof(cache_block))) == NULL)
    {
        fprintf(stderr, "allocate failed\n");
        exit(EXIT_FAILURE);
    }

    if ((new_block->content = malloc(size)) == NULL)
    {
        fprintf(stderr, "allocate failed\n");
        exit(EXIT_FAILURE);
    }

    if ((new_block->request_header = malloc(strlen(header) + 1)) == NULL) // +1 for the null terminator
    {
        fprintf(stderr, "allocate failed\n");
        exit(EXIT_FAILURE);
    }

    size_t header_size = strlen(header) + 1; // +1 for the null terminator
    strncpy(new_block->request_header, header, header_size);
    strncpy(new_block->content, content, size);

    new_block->size = size;

    // Evict LRU (Least recently used), which is the end of the list
    int shouldEvict = MAX_CACHE_SIZE < head->size + size;
    while (shouldEvict)
    {
        cache_block *tail = head->prev;

        if (tail == head)
        {
            return;
        }

        (tail->next)->prev = tail->prev;
        (tail->prev)->next = tail->next;

        head->size = head->size - tail->size;

        free(tail->content);
        free(tail->request_header);
        free(tail);
        shouldEvict = MAX_CACHE_SIZE < head->size + size;
    }
    // insert new cache entry to front of the list
    new_block->next = head->next;
    new_block->prev = head;
    // Update 2nd node's prev to new 1st block.
    (head->next)->prev = new_block;
    head->next = new_block;
    // update size in head
    head->size += size;
}

// Used for putting a recently used block to the front, as to protect it from eviction
void move_to_head(cache_block *entry)
{
    if (entry == NULL)
    {
        fprintf(stderr, "entry is null\n");
        exit(EXIT_FAILURE);
    }

    // Already at the front
    if (entry->prev == head)
        return;

    // Remove entry from its current position
    if (entry->next != NULL && entry->prev != NULL)
    {
        entry->next->prev = entry->prev;
        entry->prev->next = entry->next;
    }

    // Insert entry at the front
    entry->next = head->next;
    entry->prev = head;
    if (head->next != NULL)
        (head->next)->prev = entry;
    head->next = entry;
}

// Find first matching request header in the linked list.
// If we end up in the tail, then tail.next = head, and we are back at beginning, thus having scanned the whole list,
// and no matching requests returns null, so we can check with null on method call.
cache_block *find(char *request)
{
    cache_block *current;
    for (current = head->next; current != head; current = current->next)
    {
        if (!strcmp(request, current->request_header))
        {
            return current;
        }
    }
    return NULL;
}
