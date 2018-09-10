#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>

std::vector<int> GetRandomPath(std::vector<int> oldPath)
{
	std::vector<int> newPath;

	newPath = oldPath;

	int randPoint = rand() % (oldPath.size()-1);
	int randPoint2 = rand() % (oldPath.size()-1 - randPoint) + randPoint;

	int k=1;
	for(int i =0; i < randPoint; i++)
	{
		newPath.push_back(oldPath.at(i));
	}
	for(int i=randPoint; i < randPoint2; i++)
	{

		newPath.push_back(oldPath.at(randPoint2-k));
		k++;
	}
	for(int i=randPoint2; i < oldPath.size(); i++)
	{
		newPath.push_back(oldPath.at(i));
	}
	return newPath;
}

std::vector<int> GetRandomPathFast(std::vector<int> oldPath)
{
	std::vector<int> newPath = oldPath;

	int randPoint = rand() % (oldPath.size()-1);
	int randPoint2 = rand() % (oldPath.size()-1 - randPoint) + randPoint;

	int k = 1;
	for(int i=randPoint; i < randPoint2; i++)
	{
		newPath[i] = oldPath[randPoint2-k];
		k++;
	}
	return newPath;
}

int main() {
	std::vector<int> test;
	int n = 1000;
	for (int i = 0; i < n; i++) {
		test.push_back(i);
	}
	std::clock_t start0 = std::clock();
	for (int i = 0; i < 10000; i++) {
		std::vector<int> newPath0 = GetRandomPathFast(test);
	}
	std::clock_t end0 = std::clock();

	std::clock_t start1 = std::clock();
	for (int i = 0; i < 10000; i++) {
		std::vector<int> newPath1 = GetRandomPath(test);
	}
	std::clock_t end1 = std::clock();
	std::cout << "Time for GetRandomPath: " << (end1 - start1) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
	std::cout << "Time for GetRandomPathFast: " << (end0 - start0) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
	return 0;
}