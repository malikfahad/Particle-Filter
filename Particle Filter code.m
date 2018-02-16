clear;
%% Setting state variables and observations
n=100;
N=10000;
ESS=500;
ESS2=500;
%% Dynamics of state variables and observations
X_mean=@(x,i) x./2+25.*x./(1+x.^2)+8*cos(1.2*i);
X_val=@(x,i) normrnd(X_mean(x,i),sqrt(10));
Y_val=@(x) normrnd(x.^2./20,2);
Y_pdf=@(y,x) normpdf(y,x.^2./20,2);
Y=ones(n,1);
%%Generate the sample paths of Xt and Yt 
X_tval(1)=X_val(0,1);
Y(1)=Y_val(X_tval(1));
for i=2:n
    X_tval(i)=X_val(X_tval(i-1),i);
    Y(i)=Y_val(X_tval(i));
end
%%Treating the state variables Xt as unknowns and estimate
X=ones(n,N);
W=ones(n,N);
X2=ones(n,N);
W2=ones(n,N);
V=ones(n,N);
X(1,:)=X_val(zeros(1,N),1);
W(1,:)=Y_pdf(Y(1),X(1,:));
W(1,:)=W(1,:)/sum(W(1,:));
X2(1,:)=X_val(zeros(1,N),1);
W2(1,:)=Y_pdf(Y(1),X2(1,:));
W2(1,:)=W2(1,:)/sum(W2(1,:));

%% Particle Filter Sampling for t=1
%sequential importance resampling
if 1/sum(W(1,:).^2)<ESS
    WC=cumsum(W(1,:));
    X_temp=X(1,:);
    for j=1:N
        X(1,j)=X_temp(sum(WC<rand())+1);
    end
    W(1,:)=1/n;
end
%auxiliary particle filter sample
if 1/sum(W2(1,:).^2)<ESS
    wc2=cumsum(W2(1,:));
    X2_temp=X2(1,:);
    for j=1:N
        X2(1,j)=X2_temp(sum(wc2<rand())+1);
    end
    W2(1,:)=W2(1,:).*Y_pdf(Y(1),X_mean(X2(1,:),1));
end
  
%% Particle Filter Sampling for t>1
for i=2:n
    %%sequential importance resampling
    X(i,:)=X_val(X(i-1,:),i);
    W(i,:)=W(i-1,:).*Y_pdf(Y(i),X(i,:));
    W(i,:)=W(i,:)/sum(W(i,:));
    if 1/sum(W(i,:).^2)<ESS
        WC=cumsum(W(i,:));
        X_temp=X;
        for j=1:N
            X(:,j)=X_temp(:,sum(WC<rand())+1);
        end
        W(i,:)=1/N;
    end
    %%auxiliary particle filter sample
    X2(i,:)=X_val(X2(i-1,:),i);
    W2(i,:)=W2(i-1,:).*Y_pdf(Y(i),X2(i,:));
    W2(i,:)=W2(i,:)/sum(W2(i,:));
    if 1/sum(W2(i-1,:).^2)<ESS2
        X2_temp=X2;
        V=W2(i-1,:).*Y_pdf(Y(i),X_mean(X2(i-1,:),i));
        V=V/sum(V);
        VC=cumsum(V);
        %set the proposal weights 
        for j=1:N
            Temp_rand=sum(vc<rand())+1;
            X2(:,j)=X2_temp(:,Temp_rand);
            W_temp=W2(i-1,Temp_rand)/V(1,Temp_rand);
        end
        W2(i,:)=W_temp.*Y_pdf(Y(i),X_mean(X2(i-1,:),i));
        W2(i,:) = W2(i,:)./sum(W2(i,:));

    end

    figure(1);
hold off;
%%Using estimated density functions, compute mean and variance
plot(1:n,X_tval);
hold on;
Mean_1=X*W(i,:)';
Mean_2=X2*W2(i,:)';
S1=(X-repmat(Mean_1,1,N)).^2*W(i,:)';
S2=(X2-repmat(Mean_2,1,N)).^2*W2(i,:)';
errorbar(1:i,Mean_1(1:i,:),nthroot(S1(1:i,:),2),'r');
errorbar(1:i,Mean_2(1:i,:),nthroot(S2(1:i,:),2),'g');
legend('True Values', 'Sequential Filter', 'Auxilary Filter')
title('True and Estimated Values')
end
%% Plot containing the trajectories of all particles over time.
figure(2);
plot(1:n,X);
title('Trajectories - Sequential Importance Resampling Filter')
figure(3);
plot(1:n,X2);
title('Trajectories - Auxiliary Particle Filter')
%% Plot of the effective sample size over time.
figure(4);
plot(1:n,1./sum(W.^2,2));
legend('Sequential Filter')
title('Effective Sample Size - Sequential Particle Filter')
figure(5);
plot(1:n,1./sum(W2.^2,2));
legend('Auxilary Filter')
title('Effective Sample Size - Auxiliary Particle Filter')

