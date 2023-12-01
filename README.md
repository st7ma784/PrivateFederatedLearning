# PrivateFederatedLearning
A repo for testing a federated learning methodology with PTL 

##Plan 

The goals of this repo is to create 3 docker containers that replicate a VAE training. 

Each client step will have a private key- this dictates a model permutation. As well as a public key for un-permuting on the server side. 

The server will recieve the (permuted/encrypted) weights from the client, apply it's own aggregation of other clients and send back an encrypted set.

## Usage 
