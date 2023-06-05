
Just a rough sketch here (sorry lack of labeling is attrocious). This model uses LSTM to aggregate over time between 5 frames.
 

| class|   video |  SparsePCA  | UMAP | UMAP 3d | 
|----|----|---| ---- | ----|
|LLO|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_3_9_LLO_normalized_1.gif" width="240" /> | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_9_LLO_normalized_1_sparse.png" width="240" />  |<img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_9_LLO_normalized_1_umap.png" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_9_LLO_normalized_1_umap3d.png" width="240" /> |
|LLO|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_3_15_LLO1part1_normalized_1.gif" width="240" />  |<img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_15_LLO1part1_normalized_1_sparse.png" width="240" />  |<img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_15_LLO1part1_normalized_1_umap.png" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_15_LLO1part1_normalized_1_umap3d.png" width="240" />  |
|LLO (short)|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_3_23_LLO_1_normalized_3.gif" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_23_LLO_1_normalized_3_sparse.png" width="240" />  |<img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_23_LLO_1_normalized_3_umap.png" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_3_23_LLO_1_normalized_3_umap3d.png" width="240" />  |
|LLO - NO COLLAPSE|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_11_3_LLO_normalized_4.gif" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_3_LLO_normalized_4_sparse.png" width="240" /> |<img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_3_LLO_normalized_4_umap.png" width="240" /> | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_3_LLO_normalized_4_umap3d.png" width="240" /> |
|LLO - NO COLLAPSE|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_11_22_LLO_30nm_normalized_1.gif" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_22_LLO_30nm_normalized_1_sparse.png" width="240" />|<img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_22_LLO_30nm_normalized_1_umap.png" width="240" />| <img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_22_LLO_30nm_normalized_1_umap3d.png" width="240" />|
|MDIVI|<img src="/home/rachel/ornet/gifs/Mdivi1_7_14_2_normalized_2.gif" width="240" />  |  <img src="/home/rachel/ornet/projections/Mdivi1_7_14_2_normalized_2_sparse.png" width="240" /> | <img src="/home/rachel/ornet/projections/Mdivi1_7_14_2_normalized_2_umap.png" width="240" /> | <img src="/home/rachel/ornet/projections/Mdivi1_7_14_2_normalized_2_umap3d.png" width="240" /> |


dim reduction:
- Sparse PCA out of the box
- 2d umap has n_neighbors dialed up to 50 and min_dist=.5. So it emphasizes a global structure. You can see that similar videos have similar plots.
- 3d umap has n_neighbors set to 5. It emphasizes local structure. I'd like to explore the idea of meaningful clusters within these projections.

So clearly there is a lot of work to be done. It really seems like the model is capturing something with the collapses - Bends to the top left corner for LLO collapse, Bends to the bottom right corner for MDIVI collapse. But those without a collapse look unimpressive. I want to work more on the model - im not getting as low as a val loss as I got before (sigh, bad science, can't remember what I changed). I think the current model is relying on that collapse a little too much - the ones it fails to classify probably look like the last two llo ones (no collapse). But I'm not very dissapointed - much less brittle than identifying a car by its shadow. I want to attempt more regularization against changes in brightness, as we only need the presence/shape of protein, not its intensity.

 






This is what i get when I apply the same model and dim reduction on control samples (never seen)

| class|   video |  PCA  (transformed) |
|-|----|----|
|Control (brightening/shrinking tail)|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_11_21_normalized_10.gif" width="240" /> | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_11_21_normalized_10.png" width="240" />   
|Control (Beautiful and uneventful)|<img src="/home/rachel/ornet/gifs/DsRed2-HeLa_10_26_normalized_4.gif" width="240" />  | <img src="/home/rachel/ornet/projections/DsRed2-HeLa_10_26_normalized_4.png" width="240" /> 
|Control (a bit OOD)|<img src="/home/rachel/ornet/gifs/HeLa-DsRed2_10_13001_normalized_5.gif" width="240" />  | <img src="/home/rachel/ornet/projections/HeLa-DsRed2_10_13001_normalized_5.png" width="240" /> 
