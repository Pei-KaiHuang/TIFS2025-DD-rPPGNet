import torch

def dist(x, y):
    
    # Mean centering the data
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)

    # Calculating the Pearson correlation
    numerator = (x_mean * y_mean).sum(dim=1)
    denominator = torch.sqrt((x_mean ** 2).sum(dim=1) * (y_mean ** 2).sum(dim=1))

    # Avoid division by zero
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))

    correlation = numerator / denominator

    return 1 - correlation


def KMeans(x, n_clusters, max_iter=100, tol=0.0001):
    # Randomly initialize the centroids
    centroids = x[torch.randperm(x.size(0))[:n_clusters]]
    
    for _ in range(max_iter):
        
        # Initialize a tensor to store distances
        distances = torch.zeros(x.size(0), n_clusters)

        # Compute the distance from each point to each centroid
        for i, centroid in enumerate(centroids):
            distances[:, i] = dist(x, centroid.repeat(x.size(0), 1))
            
        # print(distances)

        # Assign each point to the centroid closest to it
        labels = distances.argmin(dim=1)


        # Compute the distance from each point to each centroid
        # dist = torch.cdist(x, centroids)

        # Assign each point to the centroid closest to it
        # labels = dist.argmin(dim=1)

        # Recompute the centroids by taking the mean of all of the points assigned to each centroid
        new_centroids = torch.stack([x[labels==i].mean(0) for i in range(n_clusters)])

        # If the centroids aren't moving much anymore, then we're done
        if (new_centroids - centroids).norm() < tol:
            break

        centroids = new_centroids

    return labels, centroids



if __name__ == "__main__":
    
    x = torch.rand(20, 300)  # replace with your data
    n_clusters = 3  # replace with the number of clusters you want
    labels, centroids = KMeans(x, n_clusters)
    
    
    print(labels)
    print(centroids)