name: Pull Request Template
description: Use this template when submitting a PR
labels: ['PR']
body: |
  ## 📝 Pull Request Summary
  
  **Type of Change** (check all that apply):
  
  - [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
  - [ ] ✨ New feature (non-breaking change which adds functionality)
  - [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
  - [ ] 📚 Documentation update (README, comments, examples)
  - [ ] 🎨 Code style update (formatting, renaming)
  - [ ] ⚡ Performance improvement
  - [ ] 🧪 Test coverage improvement
  - [ ] 🔨 Refactoring (no functional changes, just code reorganization)
  - [ ] 🛠️ CI/CD or build process changes
  - [ ] Other: ___________
  
  ---
  
  ## 🎯 Purpose
  
  **Why is this change needed?**
  
  > Describe the motivation for this change. Link to related issues if applicable.
  
  **Related Issues:**
  
  - Closes #(issue number)
  - Fixes #(issue number)
  - Part of #(issue number/milestone)
  
  ---
  
  ## 📋 Changes Made
  
  ### Files Modified
  
  | File | Change Type | Description |
  |------|-------------|-------------|
  | `path/to/file.rs` | Modified | Brief description |
  | `path/to/new_file.rs` | Added | New functionality |
  | `path/to/old_file.rs` | Removed | Deprecated feature |
  
  ### Key Changes
  
  1. **Change 1:**
     - Before: ...
     - After: ...
     - Reason: ...
  
  2. **Change 2:**
     - Details...
  
  3. **Change 3:**
     - Details...
  
  ---
  
  ## 🧪 Testing
  
  ### Tests Added
  
  ```rust
  // List new tests added
  #[test]
  fn test_new_functionality() {
      // ...
  }
  ```
  
  ### Test Results
  
  **Run the following commands and paste results:**
  
  ```bash
  # Unit tests
  cargo test --workspace --lib
  
  # Integration tests (if applicable)
  cargo test --package openmini-server --test dsa_integration_test
  
  # Clippy check
  cargo clippy --package openmini-server --lib
  
  # Format check
  cargo fmt --check
  ```
  
  **Test Output:**
  
  ```
  # Paste test results here
  ```
  
  ### Manual Testing Checklist
  
  - [ ] Tested on macOS (Apple Silicon)
  - [ ] Tested on Linux (x86_64)
  - [ ] Tested with CPU backend
  - [ ] Tested with Metal/CUDA backend (if applicable)
  - [ ] Verified no regression in existing functionality
  - [ ] Checked memory usage under load
  - [ ] Validated error handling paths
  
  ---
  
  ## 🔍 Review Guidelines
  
  ### Areas Requiring Extra Attention
  
  **Point reviewers to specific areas that need careful review:**
  
  - ⚠️ **Area 1:** `path/to/file.rs:lines X-Y` - Reason for attention
  - ⚠️ **Area 2:** ...
  
  ### Questions for Reviewers
  
  1. **Question 1:** ?
  2. **Question 2:** ?
  
  ---
  
  ## 📊 Performance Impact
  
  **Does this change affect performance?**
  
  - [ ] No performance impact
  - [ ] Improves performance (describe below)
  - [ ] May degrade performance (justify trade-off)
  
  **Benchmarks (if applicable):**
  
  | Metric | Before | After | Change |
  |--------|--------|-------|--------|
  | Latency (p99) | X ms | Y ms | ±Z% |
  | Throughput | X req/s | Y req/s | ±Z% |
  | Memory | X MB | Y MB | ±Z% |
  
  **Benchmark command:**
  
  ```bash
  cargo bench --bench benchmark_name
  ```
  
  ---
  
  ## 🔄 Migration Guide (if breaking change)
  
  **If this is a breaking change, provide migration instructions:**
  
  ### For Users
  
  ```bash
  # Step 1: Update configuration
  # Edit config/server.toml:
  # old_option = "value"  # REMOVED
  # new_option = "value"  # ADD THIS
  
  # Step 2: Update code (if API changed)
  # Old: function_old()
  # New: function_new()
  ```
  
  ### For Developers
  
  ```rust
  // Breaking API change example
  // Before:
  pub fn old_api(param: OldType) -> Result<OldOutput>
  
  // After:
  pub fn new_api(param: NewType) -> Result<NewOutput>
  ```
  
  ---
  
  ## 📚 Documentation Updates
  
  **What documentation needs to be updated?**
  
  - [ ] README.md
  - [ ] CHANGELOG.md
  - [ ] RELEASE_NOTES.md
  - [ ] Code comments (docstrings)
  - [ ] API reference (rustdoc)
  - [ ] User guide / tutorials
  - [ ] No documentation updates needed
  
  **Documentation added/updated in this PR:**
  
  - [x] Updated docstring for function X
  - [x] Added example in README section Y
  
  ---
  
  ## ✅ Checklist
  
  **Before requesting review, ensure:**
  
  #### Code Quality
  - [ ] My code follows the project's coding style guidelines
  - [ ] I have performed a self-review of my own code
  - [ ] I have commented my code, particularly in hard-to-understand areas
  - [ ] I have made corresponding changes to the documentation
  - [ ] My changes generate no new warnings (`cargo clippy`)
  - [ ] My code is properly formatted (`cargo fmt`)
  
  #### Testing
  - [ ] I have added tests that prove my fix/feature works
  - [ ] All new and existing tests pass locally
  - [ ] I have tested on multiple platforms (macOS/Linux)
  - [ ] I have considered edge cases and error conditions
  
  #### Security
  - [ ] My changes do not introduce security vulnerabilities
  - [ ] I have reviewed dependencies for known CVEs
  - [ ] Sensitive data is properly handled (no hardcoded secrets)
  
  #### Git Hygiene
  - [ ] Commits are atomic (one logical change each)
  - [ ] Commit messages follow conventional commits format
  - [ ] Branch is up-to-date with main branch
  - [ ] No merge conflicts expected
  - [ ] No unnecessary files committed (debug logs, temp files)
  
  ---
  
  ## 🎬 Screenshots/Demos (if UI change)
  
  **If this PR includes visual changes, add screenshots:**
  
  **Before:**
  ![Before](link-to-screenshot)
  
  **After:**
  ![After](link-to-screenshot)
  
  ---
  
  ## 💬 Additional Notes
  
  **Any other context, rationale, or comments for reviewers:**
  
  > Add any additional information here.
  
  ---
  
  **By submitting this PR, I confirm that:**
  
  - [ ] I have read and agree to the [Contributing Guidelines](./CONTRIBUTING.md)
  - [ ] I have licensed my contributions under the MIT License
  - [ ] I am authorized to make this contribution on behalf of the contributors listed
  
  ---
  
  **Thank you for contributing to OpenMini! 🙏**
