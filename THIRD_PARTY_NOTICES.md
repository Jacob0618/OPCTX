# Third-party and external dependency notes

The standalone implementation depends on NumPy, SciPy/HiGHS and Requests.
Their distributions retain their own licenses. No dependency source or binary
is bundled here. Gemini is accessed through the public REST API using the
user's credentials and service terms.

Research snapshots retain original imports and provenance. Some historical
entry points refer to the separate CreaMem2 research tree, including its
LongMemEval clients/prompts/memory/runner and multimodal adapters. That external
tree is not automatically vendored or relicensed by this upload. See the source
manifest for unresolved dependencies and provide appropriately licensed local
copies when reproducing those entry points.

Historical MemEye experiments refer to the official MemEye benchmark and its
baseline implementations (including Mem0-derived prompts). The local upstream
MemEye checkout includes Apache-2.0 licensing and third-party license notices;
this upload does not redistribute its dataset, images or entire vendor tree.
Follow each upstream project's licensing and benchmark usage terms.

No repository-wide license is newly assigned here. Existing copyright notices
must be preserved; repository visibility does not replace a license grant.
