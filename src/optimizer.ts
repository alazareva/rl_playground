
export interface BaseOptimizer {
    /**
     * etup for the optimizer.
     * @param optimizer_info 
     */
    optimizer_init(optimizer_info)
    /**
     *  Given weights and update g, return updated weights
     * @param weights 
     * @param g 
     */
    update_weights(weights, g)
}
