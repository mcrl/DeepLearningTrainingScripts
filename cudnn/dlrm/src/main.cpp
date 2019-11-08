#include <bits/stdc++.h>

#include "configs.h"
#include "utils.h"
#include "activation.h"
#include "embedding.h"
#include "embeddingbag.h"
#include "interaction.h"
#include "fc.h"
#include "tensor.h"
#include "data.h"
#include "timer.h"

using namespace std;

int *num_features;
float *dense_input[MAXDEV], *answer[MAXDEV];
int **sparse_input[MAXDEV], **sparse_input_bag[MAXDEV];
float *host_output[MAXDEV], *host_output_grad[MAXDEV];

Tensor *dense_in[MAXDEV], *dense_in_grad[MAXDEV], **sparse_out[MAXDEV], **sparse_out_grad[MAXDEV], *top_in[MAXDEV], *top_in_grad[MAXDEV];
IntegerTensor **sparse_in[MAXDEV], **sparse_in_bag[MAXDEV];

Tensor *botFC_z[MAXDEV][4], *botFC_a[MAXDEV][4], *botFC_z_grad[MAXDEV][4], *botFC_a_grad[MAXDEV][4];
Tensor *topFC_z[MAXDEV][3], *topFC_a[MAXDEV][3], *topFC_z_grad[MAXDEV][3], *topFC_a_grad[MAXDEV][3];
FCLayer *botFC[MAXDEV][4], *topFC[MAXDEV][3];

Embedding **embedding[MAXDEV];
EmbeddingBag **embeddingbag[MAXDEV];
ActivationLayer *sigmoid[MAXDEV], *reLU[MAXDEV];
InteractionLayer *interaction[MAXDEV];

vector<Data> train_data, test_data;

/*
 * returns (number of correct guesses, avg loss)
 */
pair<int, float> batch_inference (vector<Data>& data, int ndev) {
    float threshold = 0.5;
    int accuracy = 0, batch_size = (int) data.size();
    float loss = 0.0;

    /* Prepare input: convert data object to arrays */
    for (int i = 0; i < batch_size; i++) {
        data[i].denseToArray(dense_input[ndev] + num_dense * i);
        data[i].sparseToArray(sparse_input[ndev], i);
        data[i].sparseToBagArray(sparse_input_bag[ndev], i);
        answer[ndev][i] = data[i].res;
    }
    dense_in[ndev]->hostToDevice(dense_input[ndev], batch_size * num_dense * sizeof(float) );


    /* bot fc */
    botFC[ndev][0]->forward(dense_in[ndev], botFC_z[ndev][0]);
    reLU[ndev]->forward(botFC_z[ndev][0], botFC_a[ndev][0]);
    for (int i = 1; i < botFCLayers.size()-1; i++) {
        botFC[ndev][i]->forward(botFC_a[ndev][i-1], botFC_z[ndev][i]);
        reLU[ndev]->forward(botFC_z[ndev][i], botFC_a[ndev][i]);
    }


    /* embedding */
    for (int i = 0; i < num_sparse; i++) {
        if ( USEBAG ) sparse_in_bag[ndev][i]->hostToDevice(sparse_input_bag[ndev][i], batch_size * bag_size * sizeof(int) );
        else sparse_in[ndev][i]->hostToDevice(sparse_input[ndev][i], batch_size * sizeof(int) );
        if( USEBAG ) embeddingbag[ndev][i]->forward(sparse_in_bag[ndev][i], sparse_out[ndev][i]);
        else embedding[ndev][i]->forward(sparse_in[ndev][i], sparse_out[ndev][i]);
    }


    /* interact */
    interaction[ndev]->forward(botFC_a[ndev][botFCLayers.size()-2], sparse_out[ndev], top_in[ndev]);


    /* top fc */
    topFC[ndev][0]->forward(top_in[ndev], topFC_z[ndev][0]);
    reLU[ndev]->forward(topFC_z[ndev][0], topFC_a[ndev][0]);
    for (int i = 1; i < topFCLayers.size()-1; i++) {
        topFC[ndev][i]->forward(topFC_a[ndev][i-1], topFC_z[ndev][i]);
        if( i == topFCLayers.size()-2 ) sigmoid[ndev]->forward(topFC_z[ndev][i], topFC_a[ndev][i]);
        else reLU[ndev]->forward(topFC_z[ndev][i], topFC_a[ndev][i]);
    }
    topFC_a[ndev][topFCLayers.size()-2]->deviceToHost(host_output[ndev], batch_size * sizeof(float) );


    /* calc loss & accuracy */
    cudaDeviceSynchronize();
    for (int i = 0; i < batch_size; i++) {
        float y = answer[ndev][i], a = host_output[ndev][i];
        loss += - ( y * log(a) + (1 - y) * log(1 - a) ) / (batch_size);
        if ( (y > threshold) == (a > threshold) ) accuracy++;
    }
    cudaDeviceSynchronize();

    return make_pair(accuracy, loss);
}

void batch_train (vector<Data>& data, int ndev) {
    batch_inference (data, ndev); // forward first

    int batch_size = (int) data.size();
    
    /* Calculate initial gradient */
    cudaDeviceSynchronize();
    for (int i = 0; i < batch_size; i++) {
        float y = answer[ndev][i], a = host_output[ndev][i];
        host_output_grad[ndev][i] = ( (1 - y) / (1 - a) - y / a);
    }
    topFC_a_grad[ndev][topFCLayers.size()-2]->hostToDevice(host_output_grad[ndev], batch_size * sizeof(float) );
    cudaDeviceSynchronize();

    /* top fc backward */
    for (int i = topFCLayers.size() - 2; i >= 1; i--) {
        if ( i == topFCLayers.size() - 2 ) sigmoid[ndev]->backward(topFC_z[ndev][i], topFC_z_grad[ndev][i], topFC_a[ndev][i], topFC_a_grad[ndev][i]);
        else reLU[ndev]->backward(topFC_z[ndev][i], topFC_z_grad[ndev][i], topFC_a[ndev][i], topFC_a_grad[ndev][i]);
        topFC[ndev][i]->backward(topFC_a[ndev][i-1], topFC_a_grad[ndev][i-1], topFC_z_grad[ndev][i]);
    }
    reLU[ndev]->backward(topFC_z[ndev][0], topFC_z_grad[ndev][0], topFC_a[ndev][0], topFC_a_grad[ndev][0]);
    topFC[ndev][0]->backward(top_in[ndev], top_in_grad[ndev], topFC_z_grad[ndev][0]);


    /* interaction backward */
    interaction[ndev]->backward(botFC_a[ndev][botFCLayers.size()-2], botFC_a_grad[ndev][botFCLayers.size()-2], 
                                sparse_out[ndev], sparse_out_grad[ndev],
                                top_in[ndev], top_in_grad[ndev]);


    /* embedding backward */
    for (int i = 0; i < num_sparse; i++) {
        if ( USEBAG ) embeddingbag[ndev][i]->backward(sparse_in_bag[ndev][i], sparse_out[ndev][i], sparse_out_grad[ndev][i]);
        else embedding[ndev][i]->backward(sparse_in[ndev][i], sparse_out[ndev][i], sparse_out_grad[ndev][i]);
    }


    /* bot fc backward */
    for (int i = botFCLayers.size() - 2; i >= 1; i--) {
        reLU[ndev]->backward(botFC_z[ndev][i], botFC_z_grad[ndev][i], botFC_a[ndev][i], botFC_a_grad[ndev][i]);
        botFC[ndev][i]->backward(botFC_a[ndev][i-1], botFC_a_grad[ndev][i-1], botFC_z_grad[ndev][i]);
    }
    reLU[ndev]->backward(botFC_z[ndev][0], botFC_z_grad[ndev][0], botFC_a[ndev][0], botFC_a_grad[ndev][0]);
    botFC[ndev][0]->backward(dense_in[ndev], dense_in_grad[ndev], botFC_z_grad[ndev][0]);


    /*
     * At this moment, each GPU have calculated its own gradients.
     * Now we should reduce & gather every gradients before performing weight update.
     * Note that for embedding(bag) layers, inputs should be also gathered.
     */

    /* Reduce & gather top FC gradients */
    for (int i = 0; i < topFCLayers.size() - 1; i++) {
        FCLayer *fc = topFC[ndev][i];
        NCCL_CALL( ncclAllReduce(
            fc->d_weightDelta, fc->d_weightDelta,
            fc->inputSize * fc->outputSize, ncclFloat32, ncclSum,
            comms[ndev], streams[ndev]
        ));
        NCCL_CALL( ncclAllReduce(
            fc->biasDelta->d_mem, fc->biasDelta->d_mem,
            fc->outputSize, ncclFloat32, ncclSum,
            comms[ndev], streams[ndev]
        ));
    }

    /* Reduce & gather bot FC gradients */
    for (int i = 0; i < botFCLayers.size() - 1; i++) {
        FCLayer *fc = botFC[ndev][i];
        NCCL_CALL( ncclAllReduce(
            fc->d_weightDelta, fc->d_weightDelta,
            fc->inputSize * fc->outputSize, ncclFloat32, ncclSum,
            comms[ndev], streams[ndev]
        ));
        NCCL_CALL( ncclAllReduce(
            fc->biasDelta->d_mem, fc->biasDelta->d_mem,
            fc->outputSize, ncclFloat32, ncclSum,
            comms[ndev], streams[ndev]
        ));
    }

    /* Gather embeddingbag gradient (also need to gather input) */
    if ( USEBAG ) {
        for (int i = 0; i < num_sparse; i++) {
            EmbeddingBag *emb = embeddingbag[ndev][i];
            NCCL_CALL( ncclAllGather(emb->in, emb->gatheredIn, emb->batch_size * emb->bag_size, ncclInt32, comms[ndev], streams[ndev]) );
            NCCL_CALL( ncclAllGather(emb->delta, emb->gatheredDelta, emb->batch_size * emb->vector_size, ncclFloat32, comms[ndev], streams[ndev]) );
        }
    }
    /* Gather embedding gradient (also need to gather input) */
    else {
        for (int i = 0; i < num_sparse; i++) {
            Embedding *emb = embedding[ndev][i];
            NCCL_CALL( ncclAllGather(emb->in, emb->gatheredIn, emb->batch_size, ncclInt32, comms[ndev], streams[ndev]) );
            NCCL_CALL( ncclAllGather(emb->delta, emb->gatheredDelta, emb->batch_size * emb->vector_size, ncclFloat32, comms[ndev], streams[ndev]) );
        }
    }

    /* update top & bot FC weights */
    for (int i = 0; i < topFCLayers.size() - 1; i++) topFC[ndev][i]->update();
    for (int i = 0; i < botFCLayers.size() - 1; i++) botFC[ndev][i]->update();

    /* update embedding(bag) table */
    for (int i = 0; i < num_sparse; i++) {
        if ( USEBAG ) embeddingbag[ndev][i]->update();
        else embedding[ndev][i]->update();
    }
}

void init_network (int ndev) {
    CUDA_CALL( cudaSetDevice(ndev) );

    host_output[ndev] = (float*) malloc( batch_size / NDEV / NNODE * sizeof(float) );
    host_output_grad[ndev] = (float*) malloc( batch_size / NDEV / NNODE * sizeof(float) );
    dense_input[ndev] = (float*) malloc( batch_size / NDEV / NNODE * 13 * sizeof(float) );
    answer[ndev] = (float*) malloc( batch_size / NDEV / NNODE * sizeof(float) );
    sparse_input[ndev] = (int**) malloc( num_sparse * sizeof(int*) );
    sparse_input_bag[ndev] = (int**) malloc( num_sparse * sizeof(int*) );

    dense_in[ndev] = new Tensor(batch_size / NDEV / NNODE, botFCLayers[0], 1, 1, ndev);
    dense_in_grad[ndev] = new Tensor(batch_size / NDEV / NNODE, botFCLayers[0], 1, 1, ndev);
    sparse_in[ndev] = (IntegerTensor**) malloc( num_sparse * sizeof(IntegerTensor*) );
    sparse_in_bag[ndev] = (IntegerTensor**) malloc( num_sparse * sizeof(IntegerTensor*) );

    sparse_out[ndev] = (Tensor**) malloc( num_sparse * sizeof(Tensor*) );
    sparse_out_grad[ndev] = (Tensor**) malloc( num_sparse * sizeof(Tensor*) );

    top_in[ndev] = new Tensor(batch_size / NDEV / NNODE, topFCLayers[0], 1, 1, ndev);
    top_in_grad[ndev] = new Tensor(batch_size / NDEV / NNODE, topFCLayers[0], 1, 1, ndev);

    if ( USEBAG ) embeddingbag[ndev] = (EmbeddingBag**) malloc( num_sparse * sizeof(EmbeddingBag*) );
    else embedding[ndev] = (Embedding**) malloc( num_sparse * sizeof(Embedding*) );
    interaction[ndev] = new InteractionLayer(vector_size, num_sparse, topFCLayers[0], batch_size / NDEV / NNODE, ndev);

    sigmoid[ndev] = new ActivationLayer(ndev, "sigmoid");
    reLU[ndev] = new ActivationLayer(ndev, "relu");

    for (int i = 0; i < botFCLayers.size() - 1; i++) {
        botFC[ndev][i] = new FCLayer(botFCLayers[i], botFCLayers[i+1], batch_size / NDEV / NNODE, ndev, ndev == hostdev && mpi_world_rank == hostnode);
        botFC_z[ndev][i] = new Tensor(batch_size / NDEV / NNODE, botFCLayers[i+1], 1, 1, ndev);
        botFC_z_grad[ndev][i] = new Tensor(batch_size / NDEV / NNODE, botFCLayers[i+1], 1, 1, ndev);
        botFC_a[ndev][i] = new Tensor(batch_size / NDEV / NNODE, botFCLayers[i+1], 1, 1, ndev);
        botFC_a_grad[ndev][i] = new Tensor(batch_size / NDEV / NNODE, botFCLayers[i+1], 1, 1, ndev);
    }
    for (int i = 0; i < topFCLayers.size() - 1; i++) {
        topFC[ndev][i] = new FCLayer(topFCLayers[i], topFCLayers[i+1], batch_size / NDEV / NNODE, ndev, ndev == hostdev && mpi_world_rank == hostnode);
        topFC_z[ndev][i] = new Tensor(batch_size / NDEV / NNODE, topFCLayers[i+1], 1, 1, ndev);
        topFC_z_grad[ndev][i] = new Tensor(batch_size / NDEV / NNODE, topFCLayers[i+1], 1, 1, ndev);
        topFC_a[ndev][i] = new Tensor(batch_size / NDEV / NNODE, topFCLayers[i+1], 1, 1, ndev);
        topFC_a_grad[ndev][i] = new Tensor(batch_size / NDEV / NNODE, topFCLayers[i+1], 1, 1, ndev);
    }

    for (int i = 0; i < num_sparse; i++) {
        sparse_input[ndev][i] = (int*) malloc( batch_size / NDEV / NNODE * sizeof(int) );
        sparse_input_bag[ndev][i] = (int*) malloc( batch_size * bag_size / NDEV / NNODE  * sizeof(int) );
        sparse_in[ndev][i] = new IntegerTensor(batch_size / NDEV / NNODE, 1, 1, 1, ndev);
        sparse_in_bag[ndev][i] = new IntegerTensor(batch_size * bag_size / NDEV / NNODE, 1, 1, 1, ndev);
        sparse_out[ndev][i] = new Tensor(batch_size / NDEV / NNODE, vector_size, 1, 1, ndev);
        sparse_out_grad[ndev][i] = new Tensor(batch_size / NDEV / NNODE, vector_size, 1, 1, ndev);
        if ( USEBAG ) embeddingbag[ndev][i] = new EmbeddingBag(batch_size / NDEV / NNODE, num_features[i], bag_size, vector_size, ndev, ndev == hostdev && mpi_world_rank == hostnode);
        else embedding[ndev][i] = new Embedding(batch_size / NDEV / NNODE, num_features[i], vector_size, ndev, ndev == hostdev && mpi_world_rank == hostnode);
    }
}

int main (int argc, char **argv) {
    srand(time(NULL));

    if ( argc < 6 ) {
        fprintf(stderr, "Usage: ./dlrm <#node> <#gpu> <batch size> <epochs> <learning rate>\n");
        return 0;
    }
    NNODE = atoi(argv[1]);
    NDEV = atoi(argv[2]);
    batch_size = atoi(argv[3]);
    epochs = atoi(argv[4]);
    lr = -atof(argv[5]) / (1.0 * batch_size);

    train_batches = 300000 * 128 / batch_size;
    test_batches = 50000 * 128 / batch_size; 

    cout << "Using " << NDEV << " GPU with hostdev=" << hostdev << endl;


    /* should initiazlize omp -> cuda -> mpi -> nccl in order */
    omp_init();
    cuda_init();
    mpi_init();
    nccl_init();
    MPI_Barrier(MPI_COMM_WORLD);

    if( is_host() ) cout << "CUDA & NCCL initialization done" << endl;

    /* Load and prepare data */
    num_features = (int*) malloc( num_sparse * sizeof(int) );
    if ( mpi_world_rank == hostnode ) data_load(train_batches * batch_size, test_batches * batch_size, train_data, test_data, num_features);

    MPI_Bcast(num_features, num_sparse, MPI_FLOAT, hostnode, MPI_COMM_WORLD);

    /* Initialize Network */
    #pragma omp parallel
    {
        int ndev = omp_get_thread_num();
        CUDA_CALL( cudaSetDevice(ndev) );
        init_network(ndev);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if( is_host() ) cout << "Network initialization done" << endl;

    float ttime = 0.0;

    #pragma omp parallel
    {
        int ndev = omp_get_thread_num();
        for (int i = 0; i < botFCLayers.size() - 1; i++) {
            FCLayer *fc = botFC[ndev][i];
            NCCL_CALL( ncclBroadcast(fc->d_weight, fc->d_weight, fc->inputSize * fc->outputSize, ncclFloat32, hostdev, comms[ndev], streams[ndev]) );
            NCCL_CALL( ncclBroadcast(fc->bias->d_mem, fc->bias->d_mem, fc->outputSize, ncclFloat32, hostdev, comms[ndev], streams[ndev]) );
        }
        for (int i = 0; i < topFCLayers.size() - 1; i++) {
            FCLayer *fc = topFC[ndev][i];
            NCCL_CALL( ncclBroadcast(fc->d_weight, fc->d_weight, fc->inputSize * fc->outputSize, ncclFloat32, hostdev, comms[ndev], streams[ndev]) );
            NCCL_CALL( ncclBroadcast(fc->bias->d_mem, fc->bias->d_mem, fc->outputSize, ncclFloat32, hostdev, comms[ndev], streams[ndev]) );
        }
        for (int i = 0; i < num_sparse; i++) {
            if ( USEBAG ) {
                EmbeddingBag *emb = embeddingbag[ndev][i];
                NCCL_CALL( ncclBroadcast(emb->table, emb->table, emb->rows * emb->vector_size, ncclFloat32, hostdev, comms[ndev], streams[ndev]) );    
            }
            else {
                Embedding *emb = embedding[ndev][i];
                NCCL_CALL( ncclBroadcast(emb->table, emb->table, emb->rows * emb->vector_size, ncclFloat32, hostdev, comms[ndev], streams[ndev]) );
            }
        }
    }

    for (int ndev = 0; ndev < NDEV; ndev++) {
        CUDA_CALL( cudaSetDevice(ndev) );
        CUDA_CALL( cudaDeviceSynchronize() );
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if( is_host() ) cout << "Scattering network done" << endl;

    /////////////////////////////////////////////////////////////////////
    //                         Run Epochs                              //
    /////////////////////////////////////////////////////////////////////

    int *serialized_data = (int*) malloc( batch_size * 40 * sizeof(int) );
    for (int epoch = 0; epoch < epochs; epoch++) {
        int accuracy = 0, accuracy_sum = 0;
        float loss = 0.0, loss_sum = 0.0, iter_time = 0.0;

        /* 
         * Testing
         */
        for (int batch = 0; batch < test_batches; batch++) {
            if( is_host() ) cout << "\rTesting batch #" << batch << "/" << test_batches << std::flush;
            if( is_host() ) startTimer("test_iteration");

            /* Scatter data */
            /* Serialization is needed since we cannot directly send Data objects. */
            vector<Data> batch_data;
            if( is_host() ) { // host : read data and send to others
                for (int i = 0; i < batch_size; i++) batch_data.push_back(test_data[batch * batch_size + i]);
                for (int i = 0; i < batch_size; i++) batch_data[i].serialize( serialized_data + i * 40 );
            }

            MPI_Scatter(serialized_data, batch_size / NNODE * 40, MPI_INT, serialized_data, batch_size / NNODE * 40, MPI_INT, hostnode, MPI_COMM_WORLD);

            batch_data.resize( batch_size / NNODE );
            if( !is_host() ) { // non-host : recieve serialized data and deserialize
                for (int i = 0; i < batch_size / NNODE; i++) batch_data[i].deserialize( serialized_data + i * 40 );
            }

            /* Divide batch into mini-batches. Run mini-batch */
            vector<Data> mini_batch[4];
            pair<int, float> res[4];
            #pragma omp parallel
            {
                int ndev = omp_get_thread_num();
                for (int i = ndev * (batch_size / NNODE / NDEV); i < (ndev + 1) * (batch_size / NNODE / NDEV); i++) {
                    mini_batch[ndev].push_back(batch_data[i]);
                }
                res[ndev] = batch_inference(mini_batch[ndev], ndev);
            }

            /* Collect loss across devices */
            for (int ndev = 0; ndev < NDEV; ndev++) {
                accuracy += res[ndev].first;
                loss += res[ndev].second / test_batches / NDEV / NNODE;
            }

            /* Collect loss across nodes */
            MPI_Reduce(&accuracy, &accuracy_sum, 1, MPI_INT, MPI_SUM, hostnode, MPI_COMM_WORLD);
            MPI_Reduce(&loss, &loss_sum, 1, MPI_FLOAT, MPI_SUM, hostnode, MPI_COMM_WORLD);

            if( is_host() ) iter_time += stopTimer("test_iteration");
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if( is_host() ) {
            cout << "\r[Epoch " << epoch << "/" << epochs << "] ";
            cout << "accuracy: " << (float) accuracy_sum / ( test_batches * batch_size ) * 100.0 << "%, ";
            cout << "loss: " << loss_sum << ", " << iter_time / test_batches  * 1000.0 << "ms/test itearation\n";
        }

        if ( inference_only ) continue;


        /* 
         * Training
         */
        iter_time = 0.0;
        for (int batch = 0; batch < train_batches; batch++) {
            if ( is_host() ) cout << "\rTraining batch #" << batch << "/" << train_batches << std::flush;
            
            /* Scatter data */
            vector<Data> batch_data;
            if( is_host() ) { // host : read data and send to others
                for (int i = 0; i < batch_size; i++) batch_data.push_back(train_data[batch * batch_size + i]);
                for (int i = 0; i < batch_size; i++) batch_data[i].serialize( serialized_data + i * 40 );
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if ( is_host() ) startTimer("train_iteration");

            MPI_Scatter(serialized_data, batch_size / NNODE * 40, MPI_INT, serialized_data, batch_size / NNODE * 40, MPI_INT, hostnode, MPI_COMM_WORLD);
            batch_data.resize( batch_size / NNODE );
            if( !is_host() ) { // non-host : recieve serialized data and deserialize
                for (int i = 0; i < batch_size / NNODE; i++) batch_data[i].deserialize( serialized_data + i * 40 );
            }

            vector<Data> mini_batch[4];
            #pragma omp parallel
            {
                int ndev = omp_get_thread_num();
                for (int i = ndev * (batch_size / NNODE / NDEV); i < (ndev + 1) * (batch_size / NNODE / NDEV); i++) {
                    mini_batch[ndev].push_back(batch_data[i]);
                }
                batch_train(mini_batch[ndev], ndev);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if ( is_host() ) iter_time += stopTimer("train_iteration");
        }
        cudaDeviceSynchronize();

        if ( is_host() ) cout << "\n" << iter_time / train_batches * 1000.0 << "ms/train itearation\n";
    }

    if ( is_host() ) cout << "Total elpased time: " << (float) clock() / CLOCKS_PER_SEC << " sec\n";
}
