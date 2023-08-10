__kernel void phase_sum(__global double* q_vecs,__global double* atom_vecs, int n_atom,__global double * a2,__global double* b2){

    double real_sum = 0;
    double imag_sum = 0;

    int q_idx = get_global_id(0);

    // use q_idx to get q_x, q_y, q_z
    double q_x = q_vecs[q_idx*3];
    double q_y = q_vecs[q_idx * 3+1];
    double q_z = q_vecs[q_idx * 3+2];

    
    for (int i_atom=0; i_atom < n_atom; i_atom++){
        // use i_atom to get atom_x, atom_y, atom_z;
        double atom_x = atom_vecs[i_atom * 3]; 
        double atom_y = atom_vecs[i_atom * 3+1];
        double atom_z = atom_vecs[i_atom * 3+2];
        double phase = atom_x*q_x + atom_y*q_y + atom_z*q_z;
        double cos_term = native_cos(phase);
        double sin_term = native_sin(phase);
        real_sum += cos_term;
        imag_sum += sin_term;
    }

    a2[q_idx] += real_sum;
    b2[q_idx] += imag_sum;

}






