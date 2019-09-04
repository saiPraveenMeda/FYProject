#include<bits/stdc++.h>
using namespace std;
int main()
{
	ifstream in;
	in.open("input.txt");
	ofstream out;
	out.open("output.txt");
	string s="'";
	bool flag=false;
	while(!in.eof())
	{
		char ch;in.get(ch);
		if(ch=='\n') continue;
		if(ch==' '&&flag)
			continue;
		if(ch==' ')
		{
			s+="',";
			out<<s;
			s="'";
			flag=true;
		}
		else
		{
			flag=false;
			s.push_back(ch);
		}
	}
}