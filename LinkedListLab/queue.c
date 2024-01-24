/**
 * @file queue.c
 * @brief Implementation of a queue that supports FIFO and LIFO operations.
 *
 * This queue implementation uses a singly-linked list to represent the
 * queue elements. Each queue element stores a string value.
 *
 * Assignment for basic C skills diagnostic.
 * Developed for ITU course Operating Systems and C.
 * A fork of the C programming lab developed at CMU for courses
 * 15-213/18-213/15-513 by R. E. Bryant, 2017 (extended w/ strings 2018)
 *
 * @author JACOB GRUM jacg@itu.dk
 */

#include "queue.h"
#include "harness.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Allocates a new queue
 * @return The new queue, or NULL if memory allocation failed
 */
queue_t *queue_new(void)
{
    queue_t *q = malloc(sizeof(queue_t));
    if (q == NULL)
        return NULL;

    // There is no elements in the list, and both tail and head points to null
    q->size = 0;
    q->head = NULL;
    q->tail = NULL;
    return q;
}

/**
 * @brief Frees all memory used by a queue
 * @param[in] q The queue to free
 */
void queue_free(queue_t *q)
{
    // if empty:
    if (q == NULL)
        return;

    // Free the content of the list, before freeing the struct used for the list.
    list_ele_t *current = q->head;
    while (current != NULL)
    {
        list_ele_t *next = current->next;
        free(current->value);
        free(current);
        current = next;
    }
    // now free the struct
    free(q);
}

/**
 * @brief Attempts to insert an element at head of a queue
 *
 * This function explicitly allocates space to create a copy of `s`.
 * The inserted element points to a copy of `s`, instead of `s` itself.
 *
 * @param[in] q The queue to insert into
 * @param[in] s String to be copied and inserted into the queue
 *
 * @return true if insertion was successful
 * @return false if q is NULL, or memory allocation failed
 */
bool queue_insert_head(queue_t *q, const char *s)
{
    // if queue is uninitialized
    if (q == NULL)
        return false;

    list_ele_t *new_head;
    new_head = malloc(sizeof(list_ele_t));

    // if memory allocation failed
    if (new_head == NULL)
        return false;

    // allocate space for the string
    char *newValue = malloc((strlen(s) + 1) * sizeof(char));
    // new_head->value = malloc(strlen(s) + 1);
    if (newValue == NULL)
    {
        // free(new_head->value);
        free(new_head);
        return false;
    }
    // Sets value to the string, after having allocated space
    strcpy(newValue, s);
    new_head->value = newValue;
    // Set head of new, to old head.
    new_head->next = NULL;

    if (q->size > 0)
        new_head->next = q->head;
    else
        q->tail = new_head;
    q->size++;
    q->head = new_head;
    return true;
}

/**
 * @brief Attempts to insert an element at tail of a queue
 *
 * This function explicitly allocates space to create a copy of `s`.
 * The inserted element points to a copy of `s`, instead of `s` itself.
 *
 * @param[in] q The queue to insert into
 * @param[in] s String to be copied and inserted into the queue
 *
 * @return true if insertion was successful
 * @return false if q is NULL, or memory allocation failed
 */
bool queue_insert_tail(queue_t *q, const char *s)
{
    // If queue is uninitialized
    if (q == NULL || s == NULL)
        return false;

    list_ele_t *new_tail = malloc(sizeof(list_ele_t));

    // if memory allocation new node failed
    if (new_tail == NULL)
        return false;

    char *newValue = malloc((strlen(s) + 1) * sizeof(char));

    //  if memory allocation for string failed
    if (newValue == NULL)
    {
        free(new_tail);
        return false;
    }

    strcpy(newValue, s);
    new_tail->value = newValue;
    new_tail->next = NULL;

    // If list has no elements, set both head and tail to the same element.
    if (q->size <= 0)
    {
        q->head = q->tail;
    }
    else
    {
        q->tail->next = new_tail;
    }

    q->tail = new_tail;
    q->size++;
    return true;
}

/**
 * @brief Attempts to remove an element from head of a queue
 *
 * If removal succeeds, this function frees all memory used by the
 * removed list element and its string value before returning.
 *
 * If removal succeeds and `buf` is non-NULL, this function copies up to
 * `bufsize - 1` characters from the removed string into `buf`, and writes
 * a null terminator '\0' after the copied string.
 *
 * @param[in]  q       The queue to remove from
 * @param[out] buf     Output buffer to write a string value into
 * @param[in]  bufsize Size of the buffer `buf` points to
 *
 * @return true if removal succeeded
 * @return false if q is NULL or empty
 */
bool queue_remove_head(queue_t *q, char *buf, size_t bufsize)
{
    // false if queue is uninitialized or empty
    if (q == NULL || q->head == NULL)
    {
        return false;
    }

    list_ele_t *removed_head = q->head;

    if (buf != NULL && bufsize > 0)
    {
        // Copy the string value to buf, with a max of bufsize - 1 characters.
        strncpy(buf, removed_head->value, bufsize - 1);
        // write a null terminator '\0' after the copied string.
        buf[bufsize - 1] = '\0';
    }

    // sets new head pointer to second element
    q->head = removed_head->next;

    // free string and struct
    free(removed_head->value);
    free(removed_head);

    q->size--;
    return true;
}

/**
 * @brief Returns the number of elements in a queue
 *
 * This function runs in O(1) time.
 *
 * @param[in] q The queue to examine
 *
 * @return the number of elements in the queue, or
 *         0 if q is NULL or empty
 */
size_t queue_size(queue_t *q)
{
    // if queue is uninitialized or empty
    if (q == NULL)
        return 0;

    return q->size;
}

/**
 * @brief Reverse the elements in a queue
 *
 * This function does not allocate or free any list elements, i.e. it does
 * not call malloc or free, including inside helper functions. Instead, it
 * rearranges the existing elements of the queue.
 *
 * @param[in] q The queue to reverse
 */
void queue_reverse(queue_t *q)
{
    // If queue is empty, or only has one element, return.
    if (q == NULL || q->head == NULL || q->head->next == NULL)
        return;

    list_ele_t *current = q->head;
    list_ele_t *previous = NULL;
    list_ele_t *next = NULL;

    while (current != NULL)
    {
        // Store the next element, before reversing the link
        next = current->next;
        current->next = previous;

        // Remember last element for links
        previous = current;
        // Move to next element
        current = next;
    }
    // Update our new head and tail.
    q->tail = q->head;
    q->head = previous;
}