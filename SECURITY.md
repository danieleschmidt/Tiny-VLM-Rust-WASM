# Security Policy

## Supported Versions

We actively maintain security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### Private Disclosure
1. **Do not** create a public GitHub issue
2. Email security concerns to: [security@tinvlm.dev] (replace with actual email)
3. Include "SECURITY" in the subject line
4. Provide detailed information about the vulnerability

### Information to Include
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if known)
- Your contact information

### Response Timeline
- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 72 hours
- **Status Updates**: Weekly until resolved
- **Fix Released**: Target 7-14 days for critical issues

## Security Considerations

### Memory Safety
Tiny-VLM-Rust-WASM leverages Rust's memory safety guarantees:
- No buffer overflows in safe Rust code
- Automatic memory management prevents leaks
- Type system prevents many runtime errors

### WebAssembly Sandboxing
WASM provides additional security benefits:
- Isolated execution environment
- Limited system access by design
- Browser security model applies

### Input Validation
All inputs are validated:
- Image dimensions checked before processing
- Text length limits enforced
- Model weights verified with checksums
- Memory allocation bounds checked

### Unsafe Code Guidelines
When unsafe Rust code is necessary (SIMD intrinsics):
```rust
// Always document safety requirements
/// # Safety
/// Caller must ensure input slice length is multiple of 4
#[target_feature(enable = "neon")]
unsafe fn simd_operation(input: &[f32]) -> Vec<f32> {
    debug_assert!(input.len() % 4 == 0);
    // Implementation with safety comments
}
```

## Known Security Limitations

### Side-Channel Attacks
- Timing attacks possible through inference duration
- Memory access patterns may leak information
- Consider constant-time implementations for sensitive use cases

### Resource Exhaustion
- Large images can consume significant memory
- Long text inputs increase processing time
- Implement appropriate limits in production

### Model Integrity
- No built-in model signing/verification
- Users should verify model checksums
- Consider implementing signed models for production

## Security Best Practices

### For Users
1. **Validate Inputs**: Always validate image and text inputs
2. **Resource Limits**: Set appropriate memory and time limits
3. **Model Sources**: Only use models from trusted sources
4. **Regular Updates**: Keep library updated to latest secure version

### For Developers
1. **Code Review**: All code changes reviewed for security implications
2. **Dependency Auditing**: Regular `cargo audit` runs
3. **Fuzzing**: Continuous fuzzing of input parsing code
4. **Static Analysis**: Automated security scanning in CI

### For Production Deployment
```rust
use tiny_vlm::{FastVLM, SecurityConfig};

let config = SecurityConfig {
    max_image_size: 1024 * 1024,  // 1MB limit
    max_text_length: 1000,        // 1000 chars
    timeout_ms: 5000,             // 5 second timeout
    memory_limit_mb: 100,         // 100MB memory limit
};

let model = FastVLM::with_security_config(config)?;
```

## Incident Response

### If You Find a Vulnerability
1. Stop using the affected functionality immediately
2. Report the issue following our disclosure process
3. Apply workarounds if available
4. Update to patched version when available

### Emergency Response
For critical vulnerabilities:
- Emergency patches released within 24-48 hours
- Security advisories published immediately
- Coordinated disclosure with major users

## Security Audits

We welcome security audits from the community:
- Professional security audits conducted annually
- Bug bounty program for verified vulnerabilities
- Public audit reports published after remediation

## Contact

For security-related questions or concerns:
- Security Email: [security@tinvlm.dev]
- Encrypted Communication: PGP key available on request
- Response Time: Within 24 hours for security issues

## Acknowledgments

We thank the security researchers and community members who help keep Tiny-VLM-Rust-WASM secure.