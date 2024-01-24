/**
 * @author Jacob Grum <jacg@itu.dk>
 */

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <bits/pthreadtypes.h>
#include "cache.h"

/* The source code for the proxy is split across three files (including this one). */
#include "proxy.h" // proxy
#include "error.h" // error reporting for ^
#include "http.h"  // http-related things for ^
#include "io.h"    // io-related things for ^

// rwlock used to let in readers concurrently, but stop concurrency when writing.
static pthread_rwlock_t rwlock;

/*
Correct passing of thread arguments: Producer Consumer Model
Allocate in main
Free in thread routine
*/
void *threadWorker(void *args)
{
    int client_fd = *((int *)args);
    pthread_detach(pthread_self());
    free(args);
    handle_connection_request(client_fd);
    close(client_fd);
    return NULL;
}

int main(int argc, char **argv)
{
    int listen_fd; // fd for connection requests from clients.
    int *client_fd;
    pthread_t tid;

    // init rwlock
    if (pthread_rwlock_init(&rwlock, NULL) != 0)
    {
        fprintf(stderr, "%s: %s\n", "Init lock failed", strerror(errno));
        exit(0);
    }

    /* Check command line args for presence of a port number. */
    if (error_args_fatal(argc, argv))
    {
        exit(1);
    }

    /* Create a `socket`, `bind` it to listen address, configure it to `listen` (for connection requests). */
    init_cache();
    listen_fd = create_listen_fd(atoi(argv[1]));

    /* Handle connection requests. */
    while (1)
    {
        printf("\e[1mawaiting connection request...\e[0m\n");
        client_fd = malloc(sizeof(int)); /* alloc memory of each thread to avoid race */
        *client_fd = accept(listen_fd, (struct sockaddr *)NULL, NULL);
        pthread_create(&tid, NULL, threadWorker, (void *)client_fd);
    }

    return 0; // Indicates "no error" (although this is never reached).
}

void handle_connection_request(int client_fd)
{
    int return_cd; // return- (aka. error-) code of function calls.

    /* "Kernel, give me the fd of a connected socket for the next connection request."
       NOTE: this blocks the proxy until a connection arrives.
       https://man7.org/linux/man-pages/man2/accept.2.html (a system call) */
    if (error_accept_fatal(client_fd))
    {
        exit(1);
    }
    if (error_accept(client_fd))
    {
        return;
    }

    /* Handle (presumably, a HTTP GET) request. */
    handle_request(client_fd);

    /* "Kernel, we done handling request; close fd." (errors ignored; see man page)
       https://man7.org/linux/man-pages/man2/close.2.html (a system call) */
    return_cd = close(client_fd);
    if (error_close(return_cd))
    { /* ignore */
    }

    printf("\e[1mfinished processing request.\e[0m\n");
}

void handle_request(int client_fd)
{
    // server file descriptor
    int server_fd;

    /* String variables */
    char buf[MAX_LINE];
    char method[MAX_LINE];
    char uri[MAX_LINE];
    char version[MAX_LINE];
    char hostname[MAX_LINE];
    char path[MAX_LINE];
    char port[MAX_LINE];
    char request_hdr_to_server[MAX_LINE];

    int return_cd;
    ssize_t num_bytes;

    // Caching variables
    char whole_buffer[MAX_OBJECT_SIZE];
    char request_header_first_line[MAX_LINE];
    cache_block *cache;

    /* read HTTP Request-line */
    num_bytes = read_line(client_fd, buf);
    if (error_read(num_bytes))
    {
        return;
    }

    // Puts first line into request_header_first_line, used for looking up the cache
    strcpy(request_header_first_line, buf);

    /* print what we just read (it's not null-terminated) */
    printf("%.*s", (int)num_bytes, buf); // typeast is safe; num_bytes <= MAX_LINE
    sscanf(buf, "%s %s %s", method, uri, version);

    /* Ignore non-GET requests (your proxy is only tested on GET requests). */
    if (error_non_get(method))
    {
        return;
    }

    // Check if request is in cache
    // Adding read lock (allows for multiple readers, and writers must wait)
    pthread_rwlock_rdlock(&rwlock);
    cache = find(request_header_first_line);
    if (cache != NULL)
    {
        num_bytes = write_all(client_fd, cache->content, cache->size);
        if (error_write_client(client_fd, num_bytes))
        {
            return;
        }
        // unlock reader lock, so we instead can do a writer lock
        pthread_rwlock_unlock(&rwlock);
        // Add writer lock, so we can change the cache, by moving this item to the front.
        pthread_rwlock_wrlock(&rwlock);
        move_to_head(cache);
        // We are done writing, unlock.
        pthread_rwlock_unlock(&rwlock);
        return;
    }
    // Request was not in cache, unlock read lock.
    pthread_rwlock_unlock(&rwlock);

    /* Parse URI from GET request */
    parse_uri(uri, hostname, path, port);

    /* Set the request header */
    return_cd = set_request_header(request_hdr_to_server, hostname, path, port, client_fd);
    if (error_header(return_cd))
    {
        return;
    }

    /* Create the server fd. */
    server_fd = create_server_fd(hostname, port);
    if (error_socket_server(server_fd))
    {
        return;
    }

    /* Write the request (header) to the server. */
    return_cd = write_all(server_fd, request_hdr_to_server, strlen(request_hdr_to_server));
    if (error_write_server(server_fd, return_cd))
    {
        return;
    }

    /* Transfer the response from the server, to the client.
       (until server responds with EOF). */
    int totalSize = 0;
    do
    {
        // Num of bytes in buffer this iteration
        num_bytes = read(server_fd, buf, MAX_LINE);
        totalSize += num_bytes;
        if (error_read_server(server_fd, num_bytes))
        {
            return;
        }
        // Take the part of the page (buf) and concat it our whole_buffer which is the whole page, but only if it fits the max object size restriction
        if (totalSize < MAX_OBJECT_SIZE)
        {
            strncat(whole_buffer, buf, num_bytes);
        }
        // Write = write
        num_bytes = write_all(client_fd, buf, num_bytes);
        if (error_write_client(client_fd, num_bytes))
        {
            return;
        }
    } while (num_bytes > 0);

    // debugging
    // printf("\n There should only be 1 print:\n");
    // printf("\n request_header_first_line print here:\n %s", request_header_first_line);
    // printf("\n buf print here:\n %s", buf);
    // printf("\n whole_buffer print here:\n %s", whole_buffer);

    //  If we can fit our page into our buffer
    if (totalSize < MAX_OBJECT_SIZE)
    {
        // write cache, add a w lock
        pthread_rwlock_wrlock(&rwlock);
        // write content to cache
        insert_head(request_header_first_line, whole_buffer, totalSize);
        // unlock
        pthread_rwlock_unlock(&rwlock);
    }

    /* success; close the file descrpitor. */
    return_cd = close(server_fd);
    if (error_close_server(return_cd))
    { /* ignore */
    }
}

int create_listen_fd(int port)
{
    /* File descriptors */
    int listen_fd; // fd for connection requests from clients.

    /* Return code */
    int return_cd;

    /* Socket address (on which proxy shall listen for connection requests) (populated soon).
       https://man7.org/linux/man-pages/man3/sockaddr.3type.html */
    struct sockaddr_in listen_addr;

    printf("\e[1mcreating listen_fd\e[0m\n");

    /* Set socket address (on which proxy shall listen for connection requests). */
    set_listen_socket_address(&listen_addr, port);

    /* "Kernel, make me a socket." (for listening to client connection requests).
       https://man7.org/linux/man-pages/man2/socket.2.html (a system call) */
    listen_fd = socket(listen_addr.sin_family, SOCK_STREAM, 0);
    if (error_socket_fatal(listen_fd))
    {
        exit(1);
    }

    /* "Kernel, if you think the address I'm binding to is already in use, then
       this socket may reuse the address." (optional)
       NOTE: quality-of-life; it takes kernel ~1 min to free up an address; w/o
       this, after proxy stopped, you have to wait a bit before you can start again).
       https://man7.org/linux/man-pages/man2/setsockopt.2.html (a system call) */
    return_cd = setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int));
    if (error_socket_option(return_cd))
    { /* ignore */
    }

    /* "Kernel, bind it to this socket address" (i.e. where proxy shall listen).
       https://man7.org/linux/man-pages/man2/bind.2.html (a system call) */
    return_cd = bind(listen_fd, (struct sockaddr *)&listen_addr, sizeof(listen_addr));
    if (error_bind_fatal(return_cd))
    {
        exit(1);
    }

    /* "Kernel, oh btw, that socket? Make it passive." (it's for connection requests)
       https://man7.org/linux/man-pages/man2/listen.2.html (a system call) */
    return_cd = listen(listen_fd, LISTENQ);
    if (error_listen_fatal(return_cd))
    {
        exit(1);
    }

    printf("\e[1mlisten_fd ready\e[0m\n");

    return listen_fd;
}

/* set the proxy socket address (where it listens for connection requests). */
void set_listen_socket_address(struct sockaddr_in *listen_addr, int port)
{
    memset(listen_addr, '0', sizeof(struct sockaddr_in)); // zero out the address
    listen_addr->sin_family = AF_INET;
    listen_addr->sin_addr.s_addr = htonl(INADDR_ANY);
    listen_addr->sin_port = htons(port);
    /* NOTE: we /should/ use `getnameinfo` & `getaddrinfo` (in real world, so should you).
       with `getaddrinfo`, we get a list of potential socket addresses, and for each
       socket address in the list, we should attempt to create + bind a socket to it
       (stopping on the first successful `socket` (i.e. create)  and 'bind'). why:
        * more robust (can bind to 32-bit and 256-bit addresses, whichever server has)
        * more secure (an attacker on `cos` cannot hijack this socket by
                       binding to a more specific address than INADDR_ANY).
       instead, here we hard-code port, pick 32-bit IP addresses, and all available interfaces.
       why: because I know cos supports this, and it is simpler; `getaddrinfo` is
       intimidating for the uninitiated. (why: check out the server socket code.) */
    printf("\033[32msuccess:\033[0m set socket address of proxy.\n");
}

int create_server_fd(char *hostname, char *port)
{
    int server_fd;
    int return_cd;

    struct addrinfo *cand_ai; // pointer to heap-allocated candidate server addresses (free this!)

    /* Get list of candidate server socket addresses. */
    return_cd = get_server_socket_address_candidates(&cand_ai, hostname, port);
    if (error_address_server(return_cd))
    {
        return -1;
    }

    struct addrinfo *curr_ai; // pointer to current candidate server address in the above list.

    /* produces a socket (server_fd) bound to the first candidate address (in cand_ai)
       for which creating (resp. binding) a socket for (resp. to) it was successful. */
    for (curr_ai = cand_ai; curr_ai != NULL; curr_ai = curr_ai->ai_next)
    {
        /* "Kernel, make me a socket." (for curr_ai)
           https://man7.org/linux/man-pages/man2/socket.2.html (a system call) */
        server_fd = socket(curr_ai->ai_family, curr_ai->ai_socktype, curr_ai->ai_protocol);
        if (server_fd == -1)
            continue; // try the next ai.

        /* "Kernel, please (attempt to) connect to said socket."
           https://man7.org/linux/man-pages/man2/connect.2.html (a system call) */
        return_cd = connect(server_fd, curr_ai->ai_addr, curr_ai->ai_addrlen);
        // return_cd = connect ( server_fd, (struct sockaddr *)&curr_ai, sizeof(curr_ai) );
        if (return_cd < 0)
        {
            printf("failure connecting to socket. trying next one.\n");
        }
        if (return_cd == 0)
            break; // success

        /* couldn't bind the socket to curr_ai. try the next ai. */
        close(server_fd);
    }
    /* free up the heap-allocated linked list. */
    freeaddrinfo(curr_ai);

    /* report errors if any. */
    if (return_cd < 0)
    {
        return -1;
    }

    /* success; return the server fd. */
    return server_fd;
}

int get_server_socket_address_candidates(struct addrinfo **cand_ai, char *hostname, char *port)
{
    struct addrinfo hints_ai; // hints for proposing candidate server addresses (i.e. for generating cand_ai)
    /* set hints. network socket, numeric port, avoid IPv6 socket for hosts that don't support those. */
    memset(&hints_ai, 0, sizeof(struct addrinfo));
    hints_ai.ai_socktype = SOCK_STREAM;
    hints_ai.ai_flags = AI_NUMERICSERV | AI_ADDRCONFIG;
    return getaddrinfo(hostname, port, &hints_ai, cand_ai);
}
