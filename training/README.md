# Training Service

This service handles the fine-tuning of Gemma models using LoRA. It pulls datasets from GCS, loads the model, configures training parameters based on user input, and saves the trained adapter to GCS.

Implementation follows [Gemma Fine Tuning (Text) Guide](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora).

## Deploying to Cloud Run with GPU Support

### Prerequisites

Refer to [this documentation](https://cloud.google.com/run/docs/configuring/services/gpu) for more info. I already have the artifacts repository created for the gemma-fine-tuning project.

```bash
gcloud artifacts repositories create gemma-fine-tuning \
  --repository-format docker \
  --location us-central1
```

Set the `PROJECT_ID` and `DATA_BUCKET_NAME` and `EXPORT_BUCKET_NAME` environment variables:

```bash
export PROJECT_ID=your-project-id
export DATA_BUCKET_NAME=gemma-dataset-dev
export EXPORT_BUCKET_NAME=gemma-export-dev
```

### Build and Push the Docker Image

Build this locally (highly unrecommended):

```bash
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/training-service .

docker push us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/training-service
```

Build this on cloud build instead (recommended due to huge base image size):

```bash
gcloud builds submit --config cloudbuild.yaml \
  --project $PROJECT_ID
```

### Deploy to Cloud Run

> We are in the process of migrating this to Terraform IaC since it is easy to mess up these configurations manually. However, this process will take quite long so please use this manual deployment method for now.

Default GPU type is `--gpu-type nvidia-l4`

```bash
gcloud run deploy training-service \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/training-service \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --no-cpu-throttling \
  --max-instances 1 \
  --gpu 1 \
  --set-env-vars GCS_DATA_BUCKET_NAME=$BUCKET_NAME \
  --set-env-vars GCS_EXPORT_BUCKET_NAME=$EXPORT_BUCKET_NAME \
  --no-gpu-zonal-redundancy
```

To remove GPU: `gcloud run services update SERVICE --gpu 0`

Note that for free tier you must set no zonal redundancy otherwise it will say you don't have enough quota bla bla bla.

**After pushing a new image you can update the service with:**

```bash
gcloud run services update training-service \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/training-service
```

## TODO

- Add support for visual tasks once the preprocessing for that is implemented, see the multimodal guide

- Monitoring the training job progress using TensorBoard or w&b, I intentionally disabled the built in logging and also removed using my custom logging, will add that in the future
