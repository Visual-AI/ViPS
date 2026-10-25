# License scope and third-party notices

Original ViPS code, configuration files, scripts, tests, and documentation are
licensed under the [MIT License](LICENSE), except where a file carries an
upstream license notice or belongs to a component listed below. The root MIT
license does not replace those upstream terms or their copyright notices.

| Component | Applicable license | Upstream source |
|:--|:--|:--|
| `third_party/transformers/`, including the bundled modifications | [Apache-2.0](third_party/transformers/LICENSE), with any more specific notices retained in individual files | [Hugging Face Transformers](https://github.com/huggingface/transformers) |
| `third_party/lmms-eval/`, including the bundled modifications | [Apache-2.0](third_party/lmms-eval/LICENSE), with any more specific notices retained in individual files | [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) |
| The eight proxy layer files listed below | [Apache-2.0](vips/feature_extraction/vggt_multihead/LICENSE-APACHE), as stated in their existing headers | [VGGT](https://github.com/facebookresearch/vggt) and the sources cited in each file |
| The remaining VGGT-derived proxy code in `vips/feature_extraction/vggt_multihead/` | [VGGT License, v1, July 29, 2025](vips/feature_extraction/vggt_multihead/LICENSE) | [VGGT](https://github.com/facebookresearch/vggt) |

The Apache-2.0 proxy layer files are `attention.py`, `block.py`, `drop_path.py`,
`layer_scale.py`, `mlp.py`, `patch_embed.py`, `rope.py`, and
`vision_transformer.py`, all under
[`vips/feature_extraction/vggt_multihead/layers/`](vips/feature_extraction/vggt_multihead/layers/).
Their original copyright notices and source references are retained.

Model weights and datasets are distributed separately. This code license does
not grant a new license for those assets. Follow the terms accompanying each
checkpoint or dataset, including the applicable base-model and VGGT terms;
the model bundle must not be treated as uniformly MIT-licensed.

Checkpoint licenses are documented in the [ViPS model repository](https://huggingface.co/leeiioolin/ViPS/blob/main/LICENSE).
