# Approach note

I fine-tune the last 4 blocks of DINOv2 ViT-S/14 with a small head giving a 128-d normalised embedding. Images are resized to 224, duplicates merged, splits made by design. I train with supervised contrastive loss on batches of 32 designs x 4 views, sometimes forcing designs into one palette so colour can't separate them. Each view is recoloured, sometimes flipping light and dark, then cropped and rotated. I match by cosine similarity and accept pairs above a threshold set on validation.
