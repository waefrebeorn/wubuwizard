
#include <stdio.h>
#include "gguf_reader.h"
static const char *keys[] = {
 "general.architecture","inkling2.block_count","inkling2.embedding_length","inkling2.feed_forward_length",
 "inkling2.expert_count","inkling2.expert_used_count","inkling2.expert_feed_forward_length",
 "inkling2.attention.key_length","inkling2.num_loops","inkling2.loop.core_start","inkling2.loop.core_end",
 "inkling2.loop.gate_rank","inkling2.attention.sliding_window","inkling2.shortconv.l_cache",
 "inkling2.attn_res.block_size","inkling2.vocab_size","inkling2.vocab_size_unpadded",
 "inkling2.rope.dimension_count", NULL };
static const char *fkeys[] = {"inkling2.attention.layer_norm_rms_epsilon","inkling2.expert_weights_scale",
 "inkling2.activation.situ_beta","inkling2.activation.situ_linear_beta","inkling2.final_logit_softcapping", NULL};
int main(void){
  gguf_ctx *c = gguf_open("C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf");
  if(!c){printf("open fail\n");return 1;}
  char arch[64]={0};
  printf("arch str rc=%d val=%s\n", gguf_kv_get_str(c,"general.architecture",arch,sizeof arch), arch);
  for(int i=0; keys[i]; i++) printf("i32 %-45s rc=%d val=%d\n", keys[i], gguf_kv_get_i32(c,keys[i],NULL) , ({int v=-999;gguf_kv_get_i32(c,keys[i],&v);v;}));
  for(int i=0; fkeys[i]; i++){ float f=-999; int rc=gguf_kv_get_f32(c,fkeys[i],&f); printf("f32 %-45s rc=%d val=%g\n", fkeys[i], rc, f); }
  return 0;
}
